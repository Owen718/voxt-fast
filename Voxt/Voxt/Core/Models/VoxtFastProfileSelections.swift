// VoxtFastProfileSelections.swift
// Narrows the Voxt Fast build's model choices to the models its bundled engines accelerate.

import Foundation

/// In Voxt Fast the settings offer only the two accelerated models: other local models, GGUF
/// translation models, Apple's built-in Direct Dictation and Apple Intelligence, and every
/// remote provider are hidden, and stored selections of them fall back to the accelerated
/// models (`MLXModelCatalog.availableModels`, `CustomLLMModelCatalog.availableModels`,
/// `GGUFTranslationModelCatalog.availableModels`, `RemoteASRProvider.visibleCases`,
/// `RemoteLLMProvider.visibleCases`, `FeatureSettingsNormalization`).
extension VoxtFastProfile {
    /// Apple's built-in models (Direct Dictation, Apple Intelligence) are offered outside the profile only.
    nonisolated static var showsAppleBuiltInModels: Bool { !isActive }

    /// Points stored legacy selections that the profile hides at the accelerated models.
    /// Runs at launch before the model managers read these keys; the feature-settings snapshot
    /// itself is normalized by `FeatureSettingsNormalization`, which also rewrites most of
    /// these keys when it is saved. Keys that are not stored yet are left alone.
    @MainActor
    static func sanitizeStoredSelections(defaults: UserDefaults) {
        guard isActive else { return }

        if let repo = defaults.string(forKey: AppPreferenceKey.mlxModelRepo),
           !MLXModelCatalog.isAvailableModelRepo(repo) {
            defaults.set(asrModelRepo, forKey: AppPreferenceKey.mlxModelRepo)
        }
        for key in [
            AppPreferenceKey.customLLMModelRepo,
            AppPreferenceKey.translationCustomLLMModelRepo,
            AppPreferenceKey.rewriteCustomLLMModelRepo,
        ] {
            if let repo = defaults.string(forKey: key), !CustomLLMModelCatalog.isAvailableModelRepo(repo) {
                defaults.set(llmModelRepo, forKey: key)
            }
        }

        if let raw = defaults.string(forKey: AppPreferenceKey.transcriptionEngine),
           TranscriptionEngine.resolved(rawValue: raw) != .mlxAudio {
            defaults.set(TranscriptionEngine.mlxAudio.rawValue, forKey: AppPreferenceKey.transcriptionEngine)
        }
        if let raw = defaults.string(forKey: AppPreferenceKey.enhancementMode),
           let mode = EnhancementMode(rawValue: raw),
           mode == .appleIntelligence || mode == .remoteLLM {
            defaults.set(EnhancementMode.customLLM.rawValue, forKey: AppPreferenceKey.enhancementMode)
        }
        for key in [AppPreferenceKey.translationModelProvider, AppPreferenceKey.translationFallbackModelProvider] {
            if let raw = defaults.string(forKey: key),
               TranslationModelProvider.resolved(rawValue: raw) != .customLLM {
                defaults.set(TranslationModelProvider.customLLM.rawValue, forKey: key)
            }
        }
        if let raw = defaults.string(forKey: AppPreferenceKey.rewriteModelProvider),
           RewriteModelProvider(rawValue: raw) != .customLLM {
            defaults.set(RewriteModelProvider.customLLM.rawValue, forKey: AppPreferenceKey.rewriteModelProvider)
        }
    }
}
