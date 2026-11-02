// LocalASRModelCatalogBuilder.swift
// Provides Local ASRModel Catalog Builder for model settings.

import SwiftUI

private func localizedModelCatalog(_ key: String) -> String {
    AppLocalization.localizedString(key)
}

@MainActor
extension ModelCatalogBuilder {
    func dictationASREntry() -> ModelCatalogEntry {
        let decoration = catalogDecoration(
            base: [localizedModelCatalog("Local"), localizedModelCatalog("Built-in"), localizedModelCatalog("Fast")],
            installed: true,
            requiresConfiguration: false,
            configured: true,
            selectionID: .dictation
        )
        return ModelCatalogEntry(
            id: FeatureModelSelectionID.dictation.rawValue,
            title: localizedModelCatalog("Direct Dictation"),
            engine: localizedModelCatalog("System ASR"),
            sizeText: localizedModelCatalog("Built-in"),
            ratingText: "3.4",
            filterTags: decoration.filterTags,
            displayTags: decoration.displayTags,
            statusText: "",
            usageLocations: decoration.usageLocations,
            badgeText: nil,
            primaryAction: ModelTableAction(title: localizedModelCatalog("Settings")) {
                showASRHintTarget(.dictation)
            },
            secondaryActions: []
        )
    }

    func mlxASREntries() -> [ModelCatalogEntry] {
        mlxDisplayModelsIncludingInstalled().map { model in
            let repo = MLXModelManager.canonicalModelRepo(model.id)
            let selectionID = FeatureModelSelectionID.mlx(repo)
            let installSnapshot = mlxInstallSnapshot(repo)
            let actions = catalogActions(
                for: installSnapshot,
                removalOnly: !MLXModelCatalog.isAvailableModelRepo(repo)
            )
            let decoration = catalogDecoration(
                base: [localizedModelCatalog("Local")] + mlxCatalogTags(for: repo),
                installed: installSnapshot.isInstalled,
                requiresConfiguration: false,
                configured: true,
                selectionID: selectionID
            )

            return ModelCatalogEntry(
                id: "mlx:\(repo)",
                title: mlxModelManager.displayTitle(for: repo),
                engine: ModelCatalogEngineSupport.asrTitle(for: repo),
                sizeText: mlxASRSizeText(repo: repo, isInstalled: installSnapshot.isInstalled),
                ratingText: MLXModelManager.ratingText(for: repo),
                filterTags: decoration.filterTags,
                displayTags: decoration.displayTags,
                statusText: OmniASRBackend.unavailabilityReason(for: repo) ?? installSnapshot.statusText,
                usageLocations: decoration.usageLocations,
                badgeText: installSnapshot.badgeText ?? ModelCatalogBadgeSupport.recommendedBadgeText(forMLXRepo: repo),
                primaryAction: actions.primary,
                secondaryActions: actions.secondary
            )
        }
    }

    /// Available models are always listed. Models the catalog hides (Voxt Fast) are listed
    /// only while they still occupy disk space, so they can be removed.
    private func mlxDisplayModelsIncludingInstalled() -> [MLXModelCatalog.Option] {
        let retainedRepos = Set(MLXModelManager.supportedModels.compactMap { model -> String? in
            guard !MLXModelCatalog.isAvailableModelRepo(model.id) else { return nil }
            return Self.occupiesDisk(mlxInstallSnapshot(model.id)) ? model.id : nil
        })
        return MLXModelCatalog.displayModels(includingInstalled: retainedRepos)
    }

    private func mlxASRSizeText(repo: String, isInstalled: Bool) -> String {
        _ = isInstalled
        return mlxModelManager.remoteSizeText(repo: repo)
    }

}
