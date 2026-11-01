#[cfg(feature = "chatglm")]
pub mod chatglm;
#[cfg(feature = "deepseek")]
pub mod deepseek;
#[cfg(feature = "qwen")]
pub mod qwen;

use agent2api_provider_sdk::BoxProvider;

pub fn registry() -> Vec<BoxProvider> {
    let mut list: Vec<BoxProvider> = Vec::new();
    #[cfg(feature = "chatglm")]
    list.push(Box::new(chatglm::ChatglmProvider));
    #[cfg(feature = "deepseek")]
    list.push(Box::new(deepseek::DeepseekWebProvider));
    #[cfg(feature = "qwen")]
    list.push(Box::new(qwen::QwenProvider));
    list
}
