// VoxtFastProfileTests.swift
// Covers the Voxt Fast model profile: catalogs, provider lists and selection normalization.

import XCTest
@testable import Voxt

@MainActor
final class VoxtFastProfileTests: XCTestCase {
    private let acceleratedASRRepo = "mlx-community/Qwen3-ASR-1.7B-6bit"
    private let acceleratedLLMRepo = "mlx-community/Qwen3.5-4B-OptiQ-4bit"

    override func tearDown() {
        VoxtFastProfile.isActiveOverrideForTesting = nil
        super.tearDown()
    }

    private func withEphemeralDefaults(_ body: (UserDefaults) throws -> Void) rethrows {
        let suiteName = "VoxtFastProfileTests.\(UUID().uuidString)"
        guard let defaults = UserDefaults(suiteName: suiteName) else {
            XCTFail("Expected ephemeral UserDefaults suite")
            return
        }
        defaults.removePersistentDomain(forName: suiteName)
        defer { defaults.removePersistentDomain(forName: suiteName) }
        try body(defaults)
    }

    func testOnlyFastBundleIdentifiersActivateTheProfile() {
        XCTAssertTrue(VoxtFastProfile.isFastBundleIdentifier("com.voxt.Voxt.fast"))
        XCTAssertFalse(VoxtFastProfile.isFastBundleIdentifier("com.voxt.Voxt"))
        XCTAssertFalse(VoxtFastProfile.isFastBundleIdentifier("com.voxt.Voxt.dev"))
        XCTAssertFalse(VoxtFastProfile.isFastBundleIdentifier("com.voxt.Voxt.testhost"))
        XCTAssertFalse(VoxtFastProfile.isFastBundleIdentifier("com.example.fastapp"))
        XCTAssertFalse(VoxtFastProfile.isFastBundleIdentifier(nil))
        XCTAssertEqual(VoxtFastProfile.asrModelRepo, acceleratedASRRepo)
        XCTAssertEqual(VoxtFastProfile.llmModelRepo, acceleratedLLMRepo)
    }

    func testUnitTestHostRunsWithTheProfileInactive() {
        XCTAssertFalse(VoxtFastProfile.isActive)
    }

    func testInactiveProfileKeepsTheFullCatalogsAndProviders() {
        VoxtFastProfile.isActiveOverrideForTesting = false

        XCTAssertEqual(MLXModelCatalog.availableModels, MLXModelCatalog.supportedModels)
        XCTAssertEqual(CustomLLMModelCatalog.availableModels, CustomLLMModelCatalog.supportedModels)
        XCTAssertEqual(GGUFTranslationModelCatalog.availableModels, GGUFTranslationModelCatalog.allModels)
        XCTAssertEqual(RemoteASRProvider.visibleCases, RemoteASRProvider.allCases)
        XCTAssertEqual(RemoteLLMProvider.visibleCases, RemoteLLMProvider.allCases)
        XCTAssertTrue(VoxtFastProfile.showsAppleBuiltInModels)
    }

    func testFastProfileOffersOnlyTheAcceleratedModels() {
        VoxtFastProfile.isActiveOverrideForTesting = true

        XCTAssertEqual(MLXModelCatalog.availableModels.map(\.id), [acceleratedASRRepo])
        XCTAssertEqual(MLXModelManager.availableModels.map(\.id), [acceleratedASRRepo])
        XCTAssertEqual(CustomLLMModelCatalog.availableModels.map(\.id), [acceleratedLLMRepo])
        XCTAssertEqual(CustomLLMModelManager.availableModels.map(\.id), [acceleratedLLMRepo])
        XCTAssertTrue(GGUFTranslationModelCatalog.availableModels.isEmpty)
        XCTAssertTrue(RemoteASRProvider.visibleCases.isEmpty)
        XCTAssertTrue(RemoteLLMProvider.visibleCases.isEmpty)
        XCTAssertFalse(VoxtFastProfile.showsAppleBuiltInModels)

        // Stored repos still resolve titles; only selection is narrowed.
        XCTAssertGreaterThan(MLXModelCatalog.supportedModels.count, 1)
        XCTAssertEqual(MLXModelCatalog.displayTitle(for: "mlx-community/whisper-large-v3-turbo"), "Whisper Large v3 Turbo")
        XCTAssertTrue(CustomLLMModelCatalog.isSupportedModelRepo("mlx-community/gemma-4-e2b-it-4bit"))
        XCTAssertFalse(CustomLLMModelCatalog.isAvailableModelRepo("mlx-community/gemma-4-e2b-it-4bit"))
        XCTAssertFalse(MLXModelCatalog.isAvailableModelRepo("mlx-community/Qwen3-ASR-0.6B-4bit"))
        // Aliases of the accelerated checkpoints stay selectable.
        XCTAssertTrue(MLXModelCatalog.isAvailableModelRepo("mlx-community/Qwen3-ASR-1.7B-4bit"))
        XCTAssertTrue(CustomLLMModelCatalog.isAvailableModelRepo("mlx-community/Qwen3-4B-4bit"))
    }

    func testFastProfileListsInstalledHiddenModelsSoTheyCanBeRemoved() {
        VoxtFastProfile.isActiveOverrideForTesting = true
        let hiddenASR = "mlx-community/whisper-large-v3-turbo"
        let retiredASR = "mlx-community/GLM-ASR-Nano-2512-4bit"
        let hiddenLLM = "mlx-community/gemma-4-e2b-it-4bit"
        let retiredLLM = "mlx-community/gemma-2-2b-it-4bit"

        XCTAssertEqual(MLXModelCatalog.displayModels(includingInstalled: []).map(\.id), [acceleratedASRRepo])
        XCTAssertEqual(
            Set(MLXModelCatalog.displayModels(includingInstalled: [hiddenASR, retiredASR]).map(\.id)),
            [acceleratedASRRepo, hiddenASR]
        )
        XCTAssertEqual(CustomLLMModelCatalog.displayModels(includingInstalled: []).map(\.id), [acceleratedLLMRepo])
        XCTAssertEqual(
            Set(CustomLLMModelCatalog.displayModels(includingInstalled: [hiddenLLM, retiredLLM]).map(\.id)),
            [acceleratedLLMRepo, hiddenLLM]
        )
        XCTAssertTrue(GGUFTranslationModelCatalog.displayModels(includingInstalled: []).isEmpty)
        XCTAssertEqual(
            GGUFTranslationModelCatalog.displayModels(includingInstalled: [.hyMT2Q8_0]).map(\.id),
            [.hyMT2Q8_0]
        )

        // Installed copies are still detected, but never downloaded again.
        XCTAssertTrue(MLXModelManager.isManagedArtifactRepo(hiddenASR))
        XCTAssertFalse(MLXModelManager.isDownloadableArtifactRepo(hiddenASR))
        XCTAssertTrue(MLXModelManager.isDownloadableArtifactRepo(acceleratedASRRepo))
        XCTAssertTrue(MLXModelManager.isDownloadableArtifactRepo(SileroVADModelSupport.repo))
        XCTAssertFalse(MLXModelManager.isManagedArtifactRepo(retiredASR))
    }

    func testFastProfileNormalizesHiddenSelectionsToTheAcceleratedModels() throws {
        VoxtFastProfile.isActiveOverrideForTesting = true
        try withEphemeralDefaults { defaults in
            var settings = FeatureSettingsStore.deriveFromLegacy(defaults: defaults)
            settings.transcription.asrSelectionID = .dictation
            settings.transcription.llmEnabled = true
            settings.transcription.llmSelectionID = .appleIntelligence
            settings.transcription.notes.titleModelSelectionID = .remoteLLM(.openAI)
            settings.translation.asrSelectionID = .remoteASR(.doubaoASR)
            settings.translation.modelSelectionID = .localGGUFTranslation(.hyMT2Q4KM)
            settings.rewrite.asrSelectionID = .mlx("mlx-community/Qwen3-ASR-0.6B-4bit")
            settings.rewrite.llmSelectionID = .localLLM("mlx-community/gemma-4-e4b-it-4bit")
            settings.meeting.asrSelectionID = .mlx("mlx-community/whisper-large-v3-turbo")
            settings.meeting.summaryModelSelectionID = .remoteLLM(.anthropic)
            let data = try JSONEncoder().encode(settings)
            defaults.set(try XCTUnwrap(String(data: data, encoding: .utf8)), forKey: AppPreferenceKey.featureSettings)

            FeatureSettingsStore.migrateIfNeeded(defaults: defaults)
            let loaded = FeatureSettingsStore.load(defaults: defaults)

            let asr = FeatureModelSelectionID.mlx(acceleratedASRRepo)
            let llm = FeatureModelSelectionID.localLLM(acceleratedLLMRepo)
            XCTAssertEqual(loaded.transcription.asrSelectionID, asr)
            XCTAssertEqual(loaded.translation.asrSelectionID, asr)
            XCTAssertEqual(loaded.rewrite.asrSelectionID, asr)
            XCTAssertEqual(loaded.meeting.asrSelectionID, asr)
            XCTAssertTrue(loaded.transcription.llmEnabled)
            XCTAssertEqual(loaded.transcription.llmSelectionID, llm)
            XCTAssertEqual(loaded.transcription.notes.titleModelSelectionID, llm)
            XCTAssertEqual(loaded.translation.modelSelectionID, llm)
            XCTAssertEqual(loaded.rewrite.llmSelectionID, llm)
            XCTAssertEqual(loaded.meeting.summaryModelSelectionID, llm)

            // Legacy keys the runtime reads follow the normalized snapshot.
            XCTAssertEqual(defaults.string(forKey: AppPreferenceKey.enhancementMode), EnhancementMode.customLLM.rawValue)
            XCTAssertEqual(defaults.string(forKey: AppPreferenceKey.customLLMModelRepo), acceleratedLLMRepo)
            XCTAssertEqual(defaults.string(forKey: AppPreferenceKey.translationModelProvider), TranslationModelProvider.customLLM.rawValue)
            XCTAssertEqual(defaults.string(forKey: AppPreferenceKey.rewriteModelProvider), RewriteModelProvider.customLLM.rawValue)
        }
    }

    func testInactiveProfileKeepsBuiltInRemoteAndGGUFSelections() throws {
        VoxtFastProfile.isActiveOverrideForTesting = false
        try withEphemeralDefaults { defaults in
            var settings = FeatureSettingsStore.deriveFromLegacy(defaults: defaults)
            settings.transcription.asrSelectionID = .dictation
            settings.transcription.llmEnabled = true
            settings.transcription.llmSelectionID = .appleIntelligence
            settings.translation.asrSelectionID = .remoteASR(.doubaoASR)
            settings.translation.modelSelectionID = .localGGUFTranslation(.hyMT2Q8_0)
            settings.rewrite.asrSelectionID = .mlx("mlx-community/whisper-large-v3-turbo")
            settings.rewrite.llmSelectionID = .remoteLLM(.anthropic)
            let data = try JSONEncoder().encode(settings)
            defaults.set(try XCTUnwrap(String(data: data, encoding: .utf8)), forKey: AppPreferenceKey.featureSettings)

            let loaded = FeatureSettingsStore.load(defaults: defaults)

            XCTAssertEqual(loaded.transcription.asrSelectionID, .dictation)
            XCTAssertEqual(loaded.transcription.llmSelectionID, .appleIntelligence)
            XCTAssertEqual(loaded.translation.asrSelectionID, .remoteASR(.doubaoASR))
            XCTAssertEqual(loaded.translation.modelSelectionID, .localGGUFTranslation(.hyMT2Q8_0))
            XCTAssertEqual(loaded.rewrite.asrSelectionID, .mlx("mlx-community/whisper-large-v3-turbo"))
            XCTAssertEqual(loaded.rewrite.llmSelectionID, .remoteLLM(.anthropic))
        }
    }

    func testLaunchSanitizationPointsHiddenLegacySelectionsAtTheAcceleratedModels() {
        VoxtFastProfile.isActiveOverrideForTesting = true
        withEphemeralDefaults { defaults in
            defaults.set("mlx-community/whisper-large-v3-turbo", forKey: AppPreferenceKey.mlxModelRepo)
            defaults.set("mlx-community/gemma-4-e2b-it-4bit", forKey: AppPreferenceKey.customLLMModelRepo)
            defaults.set("mlx-community/Qwen3.5-9B-OptiQ-4bit", forKey: AppPreferenceKey.translationCustomLLMModelRepo)
            defaults.set("mlx-community/Qwen3-4B-4bit", forKey: AppPreferenceKey.rewriteCustomLLMModelRepo)
            defaults.set(TranscriptionEngine.remote.rawValue, forKey: AppPreferenceKey.transcriptionEngine)
            defaults.set(EnhancementMode.appleIntelligence.rawValue, forKey: AppPreferenceKey.enhancementMode)
            defaults.set(TranslationModelProvider.localGGUF.rawValue, forKey: AppPreferenceKey.translationModelProvider)
            defaults.set(TranslationModelProvider.remoteLLM.rawValue, forKey: AppPreferenceKey.translationFallbackModelProvider)
            defaults.set(RewriteModelProvider.remoteLLM.rawValue, forKey: AppPreferenceKey.rewriteModelProvider)

            VoxtFastProfile.sanitizeStoredSelections(defaults: defaults)

            XCTAssertEqual(defaults.string(forKey: AppPreferenceKey.mlxModelRepo), acceleratedASRRepo)
            XCTAssertEqual(defaults.string(forKey: AppPreferenceKey.customLLMModelRepo), acceleratedLLMRepo)
            XCTAssertEqual(defaults.string(forKey: AppPreferenceKey.translationCustomLLMModelRepo), acceleratedLLMRepo)
            // An alias of the accelerated checkpoint is canonicalized later, not replaced.
            XCTAssertEqual(defaults.string(forKey: AppPreferenceKey.rewriteCustomLLMModelRepo), "mlx-community/Qwen3-4B-4bit")
            XCTAssertEqual(defaults.string(forKey: AppPreferenceKey.transcriptionEngine), TranscriptionEngine.mlxAudio.rawValue)
            XCTAssertEqual(defaults.string(forKey: AppPreferenceKey.enhancementMode), EnhancementMode.customLLM.rawValue)
            XCTAssertEqual(defaults.string(forKey: AppPreferenceKey.translationModelProvider), TranslationModelProvider.customLLM.rawValue)
            XCTAssertEqual(defaults.string(forKey: AppPreferenceKey.translationFallbackModelProvider), TranslationModelProvider.customLLM.rawValue)
            XCTAssertEqual(defaults.string(forKey: AppPreferenceKey.rewriteModelProvider), RewriteModelProvider.customLLM.rawValue)
        }
    }

    func testLaunchSanitizationKeepsDisabledEnhancementAndOtherBuildsUntouched() {
        withEphemeralDefaults { defaults in
            VoxtFastProfile.isActiveOverrideForTesting = true
            defaults.set(EnhancementMode.off.rawValue, forKey: AppPreferenceKey.enhancementMode)
            VoxtFastProfile.sanitizeStoredSelections(defaults: defaults)
            XCTAssertEqual(defaults.string(forKey: AppPreferenceKey.enhancementMode), EnhancementMode.off.rawValue)
            XCTAssertNil(defaults.object(forKey: AppPreferenceKey.mlxModelRepo))
            XCTAssertNil(defaults.object(forKey: AppPreferenceKey.transcriptionEngine))

            VoxtFastProfile.isActiveOverrideForTesting = false
            defaults.set("mlx-community/whisper-large-v3-turbo", forKey: AppPreferenceKey.mlxModelRepo)
            defaults.set(TranscriptionEngine.dictation.rawValue, forKey: AppPreferenceKey.transcriptionEngine)
            VoxtFastProfile.sanitizeStoredSelections(defaults: defaults)
            XCTAssertEqual(defaults.string(forKey: AppPreferenceKey.mlxModelRepo), "mlx-community/whisper-large-v3-turbo")
            XCTAssertEqual(defaults.string(forKey: AppPreferenceKey.transcriptionEngine), TranscriptionEngine.dictation.rawValue)
        }
    }
}
