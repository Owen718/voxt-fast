// VoxtFastProfile.swift
// Identifies the Voxt Fast build, which bundles accelerated engines for two models.

import Foundation

/// Voxt Fast (a bundle identifier ending in `.fast`, see `Config/Fast.xcconfig`) bundles
/// accelerated engines for exactly one ASR checkpoint and one LLM checkpoint. Behavior that
/// differs in that build checks `isActive`; other builds are unaffected.
nonisolated enum VoxtFastProfile {
    static let bundleIdentifierSuffix = ".fast"

    /// Unit tests force the profile on or off (the test host is not a `.fast` bundle);
    /// `nil` follows the main bundle identifier.
    nonisolated(unsafe) static var isActiveOverrideForTesting: Bool?

    static var isActive: Bool {
        isActiveOverrideForTesting ?? isMainBundleFast
    }

    /// The ASR checkpoint the bundled fast engine runs.
    static var asrModelRepo: String { MLXModelCatalog.defaultModelRepo }

    /// The LLM checkpoint the bundled fast engine runs (enhancement, translation, rewrite, summaries).
    static var llmModelRepo: String { CustomLLMModelCatalog.defaultModelRepo }

    static func isFastBundleIdentifier(_ bundleIdentifier: String?) -> Bool {
        guard let bundleIdentifier = bundleIdentifier?.trimmingCharacters(in: .whitespacesAndNewlines) else {
            return false
        }
        return bundleIdentifier.hasSuffix(bundleIdentifierSuffix)
    }

    private static let isMainBundleFast = isFastBundleIdentifier(Bundle.main.bundleIdentifier)
}
