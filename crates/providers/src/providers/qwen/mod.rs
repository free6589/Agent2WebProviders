//! 通义千问网页版（`chat.qwen.ai`）：用本人网页登录态换一条 OpenAI 兼容通道。
//!
//! ── 协议依据 ────────────────────────────────────────────────
//! 2026-10 活跃开源实现（`cv-superding/dsweb-proxy` 的 `providers/qwen.ts`，
//! 其模块头注明逐行核对过 `Rfym21/Qwen2API`）：
//!   · 登录态：浏览器 localStorage 的 `token`（JWT），上游以 Cookie
//!     `token=<JWT>` 认证；
//!   · 建会话：`POST /api/v2/chats/new` → `data.id`；
//!   · 对话：`POST /api/v2/chat/completions?chat_id=<id>`，**上游只支持流式**；
//!     请求体必须带 `version: "2.1"`、`incremental_output: true`，且
//!     `messages[0].feature_config.output_schema = "phase"`（缺了不返回
//!     `delta.phase`，思考与正文就分不开）；
//!   · SSE：OpenAI chunk 形态，`delta.phase` 为 `answer` 时是正文，
//!     `think` / `thinking_summary` 时是思考；
//!   · 上游会对同一请求开多路候选（`choices[0].index` 0/1 交错），必须锁
//!     第一个出现的 index，否则正文会复读。
//!
//! ── 为什么不做自动续期 ──────────────────────────────────────
//! 网页 token 是 JWT，没有公开的 refresh 接口；过期只能在官网重新登录后
//! 再粘一次。`refresh` 能力因此不申报（与 DeepSeek 网页同一口径）。
//!
//! ── 人机验证 ────────────────────────────────────────────────
//! 上游在风控时会回 `RGV587_ERROR` / `captcha`：如实报「需要人机验证」，
//! 不尝试绕过；用户需在官方网页完成验证后更新 token。

mod prompt;

use std::time::Duration;
use md5::{Digest, Md5};
use agent2api_provider_protocol as proto;
use agent2api_provider_sdk::{control_error, ConversationSink, PluginCtx, Provider, ProviderMeta, SendFuture};
use futures::StreamExt;
use serde_json::{json, Value};

pub const ID: &str = "qwen-web";
const BASE: &str = "https://chat.qwen.ai";
/// 官方网页正常浏览器的 UA：上游对陌生 UA 会加风控，别用插件名。
const USER_AGENT: &str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36";

pub fn id(prefix: &str) -> String {
    let mut bytes = [0u8; 16];
    if getrandom::getrandom(&mut bytes).is_err() { return format!("{prefix}-{}", crate::http::now_ms()); }
    format!("{prefix}-{}", bytes.iter().map(|byte| format!("{byte:02x}")).collect::<String>())
}

/// 兜底模型目录（网页端 `/api/models` 拉不到时用）。真实目录以账号可用的为准。
fn models() -> Vec<Value> {
    ["qwen3-max", "qwen3-coder-plus", "qwen3-235b-a22b"].into_iter().map(|id| json!({
        "id": id, "name": id,
        "supportsImages": false, "supportsToolCall": true, "supportsReasoning": true,
        "toolCallMode": "emulated", "synthetic": true,
    })).collect()
}

fn client(proxy: Option<&proto::WireProxy>, gateway: &str) -> Result<reqwest::Client, String> {
    let mut builder = reqwest::Client::builder().no_proxy().redirect(reqwest::redirect::Policy::none()).connect_timeout(Duration::from_secs(15));
    let url = proxy.map(proto::WireProxy::to_url).unwrap_or_else(|| gateway.to_string());
    if !url.trim().is_empty() && url != "direct" { builder = builder.proxy(reqwest::Proxy::all(&url).map_err(|_| "千问出网代理无效")?); }
    builder.build().map_err(|error| format!("千问网络客户端创建失败：{error}"))
}

/// 网页登录态 → 请求头。`cookie` 若已是一整段 Cookie 就原样用，否则补 `token=`。
fn headers(token: &str, cookie: &str, accept: &str) -> Vec<(String, String)> {
    let cookie = if cookie.trim().is_empty() { format!("token={token}") } else { cookie.trim().to_string() };
    vec![
        ("content-type".into(), "application/json".into()),
        ("accept".into(), accept.into()),
        ("cookie".into(), cookie),
        ("user-agent".into(), USER_AGENT.into()),
        ("origin".into(), BASE.into()),
        ("referer".into(), format!("{BASE}/")),
    ]
}

async fn send_json(client: &reqwest::Client, method: &str, path: &str, headers: &[(String, String)], body: Option<Value>) -> Result<(u16, Value), String> {
    let mut request = client.request(reqwest::Method::from_bytes(method.as_bytes()).map_err(|_| "千问请求方法无效")?, format!("{BASE}{path}"))
        .timeout(Duration::from_secs(60));
    for (name, value) in headers { request = request.header(name, value); }
    if let Some(body) = body { request = request.json(&body); }
    let response = request.send().await.map_err(|error| format!("千问请求失败：{error}"))?;
    let status = response.status().as_u16();
    let bytes = response.bytes().await.map_err(|_| "千问响应读取失败")?;
    if bytes.len() > 1024 * 1024 { return Err("千问控制响应过大".into()); }
    let value: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    Ok((status, value))
}

/// 建会话 + 发对话，把上游 SSE 翻成 OpenAI chunk。
async fn forward(ctx: &PluginCtx, req: proto::ConversationRequest, sink: &mut ConversationSink<'_>) -> Result<(), String> {
    let record = ctx.store_get(ID, &req.account).await.map_err(|_| "千问网页账号不存在")?;
    let token = record.secret["accessToken"].as_str().filter(|token| !token.is_empty()).ok_or("千问网页账号缺少访问令牌")?;
    let cookie = record.secret["cookie"].as_str().unwrap_or("");
    let client = client(req.proxy.as_ref(), &ctx.env.proxy)?;
    let model = if req.wire_model.is_empty() { req.body["model"].as_str().unwrap_or("qwen3-max") } else { &req.wire_model };
    let model = if model.starts_with("qwen") { model.to_string() } else { "qwen3-max".to_string() };
    let prompt = prompt::build(&req.body)?;
    let now = crate::http::now_ms() / 1000;

    let (status, created) = send_json(&client, "POST", "/api/v2/chats/new", &headers(token, cookie, "application/json"),
        Some(json!({ "chatId": "", "models": [&model], "project_id": "", "timestamp": now, "chat_type": "t2t", "chat_mode": "normal" }))).await?;
    if !(200..300).contains(&status) { return Err(control_error(status.max(400) as i32, "千问建会话失败，请检查登录态", None, None)); }
    let chat_id = created.pointer("/data/id").and_then(Value::as_str).ok_or("千问建会话失败（没有返回 chat id）")?.to_string();

    let fid = id("fid");
    let body = json!({
        "stream": true, "version": "2.1", "incremental_output": true,
        "chat_id": &chat_id, "chatId": &chat_id, "chat_mode": "normal", "model": &model,
        "parent_id": null, "parentId": null,
        "messages": [{
            "id": null, "fid": fid, "parentId": null, "childrenIds": [], "role": "user",
            "content": prompt, "user_action": "chat", "files": [], "timestamp": now,
            "models": [&model], "model": "", "chat_type": "t2t",
            "feature_config": {
                "output_schema": "phase", "thinking_enabled": false, "research_mode": "normal",
                "auto_thinking": false, "thinking_mode": "Auto", "thinking_format": "summary",
                "auto_search": false
            },
            "extra": { "meta": { "subChatType": "t2t" } }, "sub_chat_type": "t2t"
        }],
        "timestamp": now,
    });
    let mut request = client.post(format!("{BASE}/api/v2/chat/completions?chat_id={chat_id}"))
        .json(&body).timeout(Duration::from_secs(600));
    for (name, value) in headers(token, cookie, "text/event-stream") { request = request.header(name, value); }
    let response = request.send().await.map_err(|error| format!("千问对话发送失败：{error}"))?;
    if !response.status().is_success() {
        return Err(control_error(i32::from(response.status().as_u16()), "千问对话被上游拒绝，请检查登录态或稍后重试", None, None));
    }
    if !response.headers().get("content-type").and_then(|value| value.to_str().ok()).is_some_and(|mime| mime.contains("text/event-stream")) {
        return Err(control_error(403, "千问网页要求人机验证，请在官方网页完成验证后更新 token", None, None));
    }

    let response_id = id("chatcmpl");
    let tool_mode = req.body.get("tools").and_then(Value::as_array).is_some_and(|tools| !tools.is_empty());
    let streaming = req.stream && !tool_mode;
    let mut decoder = Decoder::default();
    let mut utf8 = Vec::new();
    let mut source = response.bytes_stream();
    sink.head(proto::StreamHead { status: 200, headers: [("content-type".into(), "text/event-stream".into())].into_iter().collect(), upstream: proto::UpstreamKind::Chat }).await?;
    while let Some(bytes) = source.next().await {
        let bytes = bytes.map_err(|_| "千问对话传输中断")?;
        let before_content = decoder.content.len();
        let before_reasoning = decoder.reasoning.len();
        utf8.extend_from_slice(&bytes);
        let valid = match std::str::from_utf8(&utf8) { Ok(text) => text.len(), Err(error) if error.error_len().is_none() => error.valid_up_to(), Err(_) => return Err("千问响应不是 UTF-8".into()) };
        let text = std::str::from_utf8(&utf8[..valid]).map_err(|_| "千问响应无法解码")?.replace("\r\n", "\n");
        decoder.feed(&text)?; utf8.drain(..valid);
        if decoder.content.len() + decoder.reasoning.len() > 8 * 1024 * 1024 { return Err("千问响应超过大小上限".into()); }
        if streaming {
            let mut delta = json!({});
            if decoder.content.len() > before_content { delta["content"] = json!(&decoder.content[before_content..]); }
            if decoder.reasoning.len() > before_reasoning { delta["reasoning_content"] = json!(&decoder.reasoning[before_reasoning..]); }
            if delta.as_object().is_some_and(|value| !value.is_empty()) { send_delta(sink, &response_id, &model, delta, None).await?; }
        }
    }
    if !utf8.is_empty() { return Err("千问响应被截断".into()); }
    decoder.finish()?;
    if streaming {
        send_delta(sink, &response_id, &model, json!({}), Some("stop")).await?;
        if decoder.total_tokens > 0 {
            sink.send(format!("data: {}\n\n", json!({ "id": response_id, "object": "chat.completion.chunk", "model": model, "choices": [], "usage": { "total_tokens": decoder.total_tokens } })).as_bytes()).await?;
        }
        sink.send(b"data: [DONE]\n\n").await?;
    } else {
        let calls = prompt::parse_calls(&decoder.content, &req.body)?;
        let mut message = json!({ "role": "assistant", "content": decoder.content });
        if !decoder.reasoning.is_empty() { message["reasoning_content"] = json!(decoder.reasoning); }
        let finish = if let Some(calls) = calls { message["content"] = Value::Null; message["tool_calls"] = json!(calls); "tool_calls" } else { "stop" };
        let mut delta = message;
        if let Some(calls) = delta.get_mut("tool_calls").and_then(Value::as_array_mut) { for (index, call) in calls.iter_mut().enumerate() { call["index"] = json!(index); } }
        send_delta(sink, &response_id, &model, delta, None).await?;
        send_delta(sink, &response_id, &model, json!({}), Some(finish)).await?;
        if decoder.total_tokens > 0 {
            sink.send(format!("data: {}\n\n", json!({ "id": response_id, "object": "chat.completion.chunk", "model": model, "choices": [], "usage": { "total_tokens": decoder.total_tokens } })).as_bytes()).await?;
        }
        sink.send(b"data: [DONE]\n\n").await?;
    }
    Ok(())
}

async fn send_delta(sink: &mut ConversationSink<'_>, id: &str, model: &str, delta: Value, finish: Option<&str>) -> Result<(), String> {
    sink.send(format!("data: {}\n\n", json!({ "id": id, "object": "chat.completion.chunk", "created": crate::http::now_ms() / 1000, "model": model,
        "choices": [{ "index": 0, "delta": delta, "finish_reason": finish }] })).as_bytes()).await
}

#[derive(Default)]
struct Decoder {
    buffer: String,
    pub content: String,
    pub reasoning: String,
    pub total_tokens: u64,
    first_index: Option<String>,
    finished: bool,
}

impl Decoder {
    fn feed(&mut self, text: &str) -> Result<(), String> {
        self.buffer.push_str(text);
        if self.buffer.len() > 1024 * 1024 { return Err("千问 SSE 帧超过大小上限".into()); }
        while let Some(index) = self.buffer.find('\n') {
            let line = self.buffer[..index].to_string();
            self.buffer.drain(..index + 1);
            let line = line.trim_end_matches('\r');
            let Some(payload) = line.strip_prefix("data:") else { continue };
            let payload = payload.trim();
            if payload.is_empty() { continue; }
            if payload == "[DONE]" { self.finished = true; continue; }
            let Ok(frame) = serde_json::from_str::<Value>(payload) else { continue };
            self.frame(&frame)?;
        }
        Ok(())
    }

    fn finish(&mut self) -> Result<(), String> {
        if !self.buffer.trim().is_empty() {
            let rest = std::mem::take(&mut self.buffer);
            if let Some(payload) = rest.trim().strip_prefix("data:") {
                let payload = payload.trim();
                if payload == "[DONE]" { self.finished = true; }
                else if let Ok(frame) = serde_json::from_str::<Value>(payload) { self.frame(&frame)?; }
            }
        }
        Ok(())
    }

    fn frame(&mut self, frame: &Value) -> Result<(), String> {
        if frame.get("code").and_then(Value::as_str) == Some("RateLimited") {
            return Err("千问网页版今日配额已耗尽（RateLimited）".into());
        }
        if frame.get("code").and_then(Value::as_str) == Some("RGV587_ERROR")
            || frame.get("detail").and_then(Value::as_str).is_some_and(|detail| detail.contains("captcha")) {
            return Err("千问要求人机验证：请在官方网页完成验证后更新 token".into());
        }
        if frame.get("error").is_some_and(|error| !error.is_null()) {
            return Err(format!("千问返回错误：{}", frame["error"].to_string().chars().take(200).collect::<String>()));
        }
        if let Some(usage) = frame.get("usage") {
            let total = usage.get("total_tokens").and_then(Value::as_u64)
                .or_else(|| Some(usage.get("input_tokens")?.as_u64()? + usage.get("output_tokens")?.as_u64()?));
            if let Some(total) = total { self.total_tokens = total; }
        }
        let Some(choice) = frame.pointer("/choices/0") else { return Ok(()) };
        let index = choice.get("index").map(|value| value.to_string()).unwrap_or_else(|| "0".into());
        match &self.first_index {
            None => self.first_index = Some(index.clone()),
            Some(first) if first != &index => return Ok(()),
            _ => {}
        }
        let Some(delta) = choice.get("delta") else { return Ok(()) };
        let Some(text) = delta.get("content").and_then(Value::as_str).filter(|text| !text.is_empty()) else { return Ok(()) };
        match delta.get("phase").and_then(Value::as_str).unwrap_or("answer") {
            "think" | "thinking_summary" => self.reasoning.push_str(text),
            _ => self.content.push_str(text),
        }
        Ok(())
    }
}

impl Provider for QwenProvider {
    fn meta(&self) -> ProviderMeta {
        ProviderMeta::new(ID, "通义千问 网页")
            .with_capabilities(proto::Capabilities { chat: true, stateful: true, ..Default::default() })
            .with_models(models())
            .with_login(vec![proto::LoginMethod::Token {
                label: "网页访问令牌".into(),
                placeholder: "chat.qwen.ai 登录后 localStorage 的 token（eyJ 开头，不是 API Key）".into(),
                extra_fields: vec![],
            }])
            .with_account_features(proto::AccountFeatures { identifier: vec!["uid".into()], ..Default::default() })
    }

    fn conversation<'a>(&'a self, ctx: &'a PluginCtx, req: proto::ConversationRequest, sink: ConversationSink<'a>) -> SendFuture<'a, ()> {
        Box::pin(async move {
            let mut sink = sink;
            match forward(ctx, req, &mut sink).await {
                Ok(()) => Ok(()),
                Err(error) => {
                    let failure = error.strip_prefix("CONTROL_ERROR:").and_then(|raw| serde_json::from_str::<Value>(raw).ok());
                    if let Some(failure) = failure {
                        let status = failure["status"].as_u64().filter(|status| (400..600).contains(status)).unwrap_or(502) as u16;
                        sink.head(proto::StreamHead { status, headers: [("content-type".into(), "application/json".into())].into_iter().collect(), upstream: proto::UpstreamKind::Chat }).await?;
                        sink.send(json!({ "message": failure["message"] }).to_string().as_bytes()).await?;
                        Ok(())
                    } else { Err(error) }
                }
            }
        })
    }

    fn ensure_token<'a>(&'a self, ctx: &'a PluginCtx, account: &'a str) -> SendFuture<'a, String> {
        Box::pin(async move {
            ctx.store_get(ID, account).await.map_err(|error| error.to_string())?.secret["accessToken"]
                .as_str().map(str::to_string).ok_or_else(|| "千问网页缺少访问令牌".into())
        })
    }

    fn models<'a>(&'a self, ctx: &'a PluginCtx, account: &'a str) -> SendFuture<'a, proto::ModelList> {
        Box::pin(async move {
            let fallback = proto::ModelList { models: models(), meta: Default::default() };
            let Ok(record) = ctx.store_get(ID, account).await else { return Ok(fallback) };
            let token = record.secret["accessToken"].as_str().unwrap_or("");
            if token.is_empty() { return Ok(fallback); }
            let cookie = record.secret["cookie"].as_str().unwrap_or("");
            let client = client(None, &ctx.env.proxy)?;
            let (status, value) = send_json(&client, "GET", "/api/models", &headers(token, cookie, "application/json"), None).await?;
            if !(200..300).contains(&status) { return Ok(fallback); }
            let list = value.get("data").and_then(Value::as_array).cloned().unwrap_or_default();
            let models: Vec<Value> = list.iter().filter_map(|item| {
                let id = item.get("id").and_then(Value::as_str)?;
                Some(json!({ "id": id, "name": item.get("name").and_then(Value::as_str).unwrap_or(id),
                    "supportsImages": false, "supportsToolCall": true, "supportsReasoning": true,
                    "toolCallMode": "emulated", "synthetic": true }))
            }).collect();
            Ok(if models.is_empty() { fallback } else { proto::ModelList { models, meta: Default::default() } })
        })
    }

    fn classify(&self, status: u16, body: &Value) -> proto::ErrorClass {
        let message = body["message"].as_str().unwrap_or("千问网页请求失败").to_string();
        match status {
            401 => proto::ErrorClass::TokenExpired { message },
            429 => proto::ErrorClass::QuotaLimited { reset_at: None, message, upstream_code: None, status },
            _ => proto::ErrorClass::Fatal { status, message, upstream_code: None },
        }
    }

    fn normalize_account<'a>(&'a self, ctx: &'a PluginCtx, input: proto::NormalizeInput) -> SendFuture<'a, proto::NormalizedAccount> {
        Box::pin(async move {
            let payload = match input.source {
                proto::NormalizeSource::Manual { payload } => payload,
                proto::NormalizeSource::FileText { content, .. } => serde_json::from_str(&content).unwrap_or_else(|_| json!({ "token": content })),
                _ => return Err("千问网页请粘贴本人登录令牌（网页 localStorage 的 token）".into()),
            };
            let token = ["accessToken", "token", "apiKey"].into_iter().find_map(|key| payload[key].as_str()).unwrap_or("").trim().trim_start_matches("Bearer ").to_string();
            let cookie = payload["cookie"].as_str().unwrap_or("").trim().to_string();
            if token.is_empty() || token.len() > 16384 || token.starts_with("sk-") || cookie.len() > 32768 || token.contains(['\r', '\n']) || cookie.contains(['\r', '\n']) {
                return Err("请填写有效的千问网页访问令牌（eyJ 开头的 JWT，不是 API Key）".into());
            }
            let client = client(None, &ctx.env.proxy)?;
            let (status, value) = send_json(&client, "GET", "/api/v1/auths/", &headers(&token, &cookie, "application/json"), None).await?;
            if !(200..300).contains(&status) {
                return Err(control_error(if status == 401 { 401 } else { 400 }, "千问登录态校验失败：请在 chat.qwen.ai 重新登录后复制新的 token", None, None));
            }
            let name = value.get("name").or_else(|| value.get("email")).and_then(Value::as_str).unwrap_or("");
            let tail: String = token.chars().rev().take(8).collect::<Vec<_>>().into_iter().rev().collect();
            let uid = format!("{:x}", Md5::digest(token.as_bytes()));
            Ok(proto::NormalizedAccount {
                identity: format!("{ID}-{uid}"),
                secret: json!({ "accessToken": token, "cookie": cookie }),
                public: json!({
                    "name": input.name.unwrap_or_else(|| if name.is_empty() { format!("千问 {tail}") } else { name.to_string() }),
                    "tokenTail": tail, "uid": uid, "credentialIdentity": true
                }),
                desktop: false,
            })
        })
    }
}

pub struct QwenProvider;
