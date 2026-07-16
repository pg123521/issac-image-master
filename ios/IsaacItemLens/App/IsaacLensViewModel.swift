import Foundation
import OSLog
import PhotosUI
import SwiftUI
import UIKit

@MainActor
final class IsaacLensViewModel: ObservableObject {
  private static let candidateDisplayLimitKey = "candidateDisplayLimit"
  private static let candidateSearchLimit = 50
  private let logger = Logger(subsystem: "com.pg123521.IsaacItemLens", category: "pipeline")
  @Published var image: UIImage?
  @Published var regions: [DetectionRegion] = []
  @Published var selectedRegionID: UUID?
  @Published var matches: [SearchMatch] = []
  @Published var selectedItem: IsaacObject?
  @Published var manualMode = false
  @Published var manualBoxFraction = 0.082
  @Published var candidateDisplayLimit: Int {
    didSet {
      UserDefaults.standard.set(candidateDisplayLimit, forKey: Self.candidateDisplayLimitKey)
    }
  }
  @Published var showManualGuidance = false
  @Published var isSearching = false
  @Published var status = "上传截图以识别物品"

  let repository: ItemRepository?
  private let searchEngine: ItemSearchEngine?
  private var selectionTask: Task<Void, Never>?
  private var thumbnailCache: [UUID: UIImage] = [:]
  private var currentZoomScale: CGFloat = 1
  private var selectedRegionNeedsSearch = false

  var selectedRegion: DetectionRegion? {
    regions.first { $0.id == selectedRegionID }
  }

  init() {
    let savedCandidateLimit = UserDefaults.standard.object(forKey: Self.candidateDisplayLimitKey) as? Int ?? 15
    candidateDisplayLimit = min(max(savedCandidateLimit, 1), Self.candidateSearchLimit)
    var loadedRepository: ItemRepository?
    var loadedSearchEngine: ItemSearchEngine?
    var initialStatus = "上传截图以识别物品"
    do {
      let repository = try ItemRepository()
      loadedRepository = repository
      loadedSearchEngine = try ItemSearchEngine(repository: repository)
    } catch {
      logger.error("initialization failed error=\(error.localizedDescription, privacy: .public)")
      loadedRepository = nil
      loadedSearchEngine = nil
      initialStatus = error.localizedDescription
    }
    repository = loadedRepository
    searchEngine = loadedSearchEngine
    status = initialStatus
  }

  func importPhoto(_ item: PhotosPickerItem?) async {
    guard let item else { return }
    do {
      guard let data = try await item.loadTransferable(type: Data.self), let image = UIImage(data: data) else {
        throw ImportError.invalidImage
      }
      begin(image)
    } catch {
      status = error.localizedDescription
    }
  }

  func begin(_ newImage: UIImage) {
    selectionTask?.cancel()
    image = newImage
    regions = []
    selectedRegionID = nil
    matches = []
    selectedItem = nil
    thumbnailCache = [:]
    currentZoomScale = 1
    selectedRegionNeedsSearch = false
    manualMode = true
    showManualGuidance = true
    status = "点击图片中物品进行手动选取"
    Task { [weak self] in
      try? await Task.sleep(for: .seconds(4))
      self?.showManualGuidance = false
    }
  }

  func clearImage() {
    selectionTask?.cancel()
    image = nil
    regions = []
    selectedRegionID = nil
    matches = []
    selectedItem = nil
    thumbnailCache = [:]
    currentZoomScale = 1
    selectedRegionNeedsSearch = false
    manualMode = false
    showManualGuidance = false
    status = "上传截图以识别物品"
  }

  func closeSelection() {
    selectionTask?.cancel()
    selectedRegionID = nil
    matches = []
    selectedItem = nil
    isSearching = false
    status = "点击图片中物品进行手动选取"
  }

  func addManualRegion(at point: CGPoint, zoomScale: CGFloat) {
    guard manualMode, let image else { return }
    currentZoomScale = max(1, zoomScale)
    let pixelWidth = image.size.width * image.scale
    let pixelHeight = image.size.height * image.scale
    let side = min(pixelWidth, pixelHeight) * manualBoxFraction / currentZoomScale
    let width = min(1, side / pixelWidth)
    let height = min(1, side / pixelHeight)
    let rect = CGRect(
      x: min(max(0, point.x - width / 2), 1 - width),
      y: min(max(0, point.y - height / 2), 1 - height),
      width: width,
      height: height
    )
    let region = DetectionRegion(rect: rect)
    regions.append(region)
    showManualGuidance = false
    select(region)
    status = "已添加手动选取"
  }

  func updateRegionsForZoom(from oldScale: CGFloat, to newScale: CGFloat) {
    guard oldScale > 0, newScale > 0, oldScale != newScale else { return }
    currentZoomScale = newScale
    let sizeRatio = oldScale / newScale
    for index in regions.indices {
      let rect = regions[index].rect
      let width = min(1, max(1 / CGFloat(max(1, image?.size.width ?? 1)), rect.width * sizeRatio))
      let height = min(1, max(1 / CGFloat(max(1, image?.size.height ?? 1)), rect.height * sizeRatio))
      regions[index].rect = CGRect(
        x: min(max(0, rect.midX - width / 2), 1 - width),
        y: min(max(0, rect.midY - height / 2), 1 - height),
        width: width,
        height: height
      )
    }
    thumbnailCache = [:]
    if let selectedRegion {
      matches = []
      search(selectedRegion)
    }
  }

  func setManualBoxFraction(_ value: Double) {
    manualBoxFraction = value
    guard resizeSelectedRegionToManualBoxSize() else { return }
    selectedRegionNeedsSearch = true
  }

  func finishManualBoxResize() {
    guard selectedRegionNeedsSearch else { return }
    selectedRegionNeedsSearch = false
    if let selectedRegion {
      matches = []
      search(selectedRegion)
    }
  }

  @discardableResult
  private func resizeSelectedRegionToManualBoxSize() -> Bool {
    guard let selectedRegionID,
          let index = regions.firstIndex(where: { $0.id == selectedRegionID }),
          let image else { return false }
    let pixelWidth = image.size.width * image.scale
    let pixelHeight = image.size.height * image.scale
    let side = min(pixelWidth, pixelHeight) * manualBoxFraction / max(1, currentZoomScale)
    let width = min(1, side / pixelWidth)
    let height = min(1, side / pixelHeight)
    let rect = regions[index].rect
    regions[index].rect = CGRect(
      x: min(max(0, rect.midX - width / 2), 1 - width),
      y: min(max(0, rect.midY - height / 2), 1 - height),
      width: width,
      height: height
    )
    thumbnailCache.removeValue(forKey: selectedRegionID)
    return true
  }

  func select(_ region: DetectionRegion) {
    selectedRegionID = region.id
    selectedItem = nil
    matches = []
    search(region)
  }

  func remove(_ region: DetectionRegion) {
    regions.removeAll { $0.id == region.id }
    thumbnailCache.removeValue(forKey: region.id)
    if selectedRegionID == region.id {
      selectedRegionID = regions.first?.id
      matches = []
      selectedItem = nil
      if let first = regions.first { search(first) }
    }
    status = regions.isEmpty ? "未选中区域，可手动选取" : "已删除选中区域"
  }

  func thumbnail(for region: DetectionRegion) -> UIImage? {
    if let cached = thumbnailCache[region.id] { return cached }
    guard let image, let source = try? ImageUtilities.normalizedCGImage(image),
          let crop = try? ImageUtilities.squareCrop(source, normalizedRect: region.rect),
          let buffer = try? ImageUtilities.searchInputBuffer(from: crop),
          let thumbnail = try? ImageUtilities.image(from: buffer) else { return nil }
    thumbnailCache[region.id] = thumbnail
    return thumbnail
  }

  private func search(_ region: DetectionRegion) {
    selectionTask?.cancel()
    guard let searchEngine, let image else { return }
    isSearching = true
    selectionTask = Task { [weak self] in
      do {
        let result = try await searchEngine.search(
          image: image,
          region: region,
          topK: Self.candidateSearchLimit
        )
        guard !Task.isCancelled, self?.selectedRegionID == region.id else { return }
        self?.thumbnailCache[region.id] = result.modelInputImage
        self?.matches = result.matches
        self?.isSearching = false
        self?.logger.info(
          "UI preview uses exact model input region=\(region.id.uuidString, privacy: .public) \(result.modelInputHash, privacy: .public) saved=\(result.savedInputURL.path, privacy: .public)"
        )
      } catch {
        guard !Task.isCancelled else { return }
        self?.logger.error("search failed region=\(region.id.uuidString, privacy: .public) error=\(error.localizedDescription, privacy: .public)")
        self?.matches = []
        self?.isSearching = false
        self?.status = error.localizedDescription
      }
    }
  }
}

enum ImportError: LocalizedError {
  case invalidImage

  var errorDescription: String? { "无法读取这张图片" }
}
