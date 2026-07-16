import Foundation
import OSLog
import PhotosUI
import SwiftUI
import UIKit

@MainActor
final class IsaacLensViewModel: ObservableObject {
  private static let autoDetectionKey = "autoDetectionEnabled"
  private let logger = Logger(subsystem: "com.pg123521.IsaacItemLens", category: "pipeline")
  @Published var image: UIImage?
  @Published var regions: [DetectionRegion] = []
  @Published var selectedRegionID: UUID?
  @Published var matches: [SearchMatch] = []
  @Published var selectedItem: IsaacObject?
  @Published var stage: DetectionStage = .idle
  @Published var progress: Double = 0
  @Published var manualMode = false
  @Published var manualBoxFraction = 0.082
  @Published var autoDetectionEnabled: Bool {
    didSet {
      UserDefaults.standard.set(autoDetectionEnabled, forKey: Self.autoDetectionKey)
    }
  }
  @Published var showManualGuidance = false
  @Published var isSearching = false
  @Published var status = "上传截图以识别物品"

  let repository: ItemRepository?
  private let detector: CollectibleDetector?
  private let searchEngine: ItemSearchEngine?
  private var selectionTask: Task<Void, Never>?
  private var detectionTask: Task<Void, Never>?

  var selectedRegion: DetectionRegion? {
    regions.first { $0.id == selectedRegionID }
  }

  init() {
    autoDetectionEnabled = UserDefaults.standard.object(forKey: Self.autoDetectionKey) as? Bool ?? true
    var loadedRepository: ItemRepository?
    var loadedDetector: CollectibleDetector?
    var loadedSearchEngine: ItemSearchEngine?
    var initialStatus = "上传截图以识别物品"
    do {
      let repository = try ItemRepository()
      loadedRepository = repository
      loadedDetector = try CollectibleDetector()
      loadedSearchEngine = try ItemSearchEngine(repository: repository)
    } catch {
      logger.error("initialization failed error=\(error.localizedDescription, privacy: .public)")
      loadedRepository = nil
      loadedDetector = nil
      loadedSearchEngine = nil
      initialStatus = error.localizedDescription
    }
    repository = loadedRepository
    detector = loadedDetector
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
      stage = .failed(error.localizedDescription)
      progress = 1
      status = error.localizedDescription
    }
  }

  func begin(_ newImage: UIImage) {
    detectionTask?.cancel()
    selectionTask?.cancel()
    image = newImage
    regions = []
    selectedRegionID = nil
    matches = []
    selectedItem = nil
    manualMode = !autoDetectionEnabled
    showManualGuidance = false
    if autoDetectionEnabled {
      startDetection()
    } else {
      stage = .idle
      progress = 0
      showManualGuidance = true
      status = "自动检测已关闭，请手动选取"
    }
  }

  func toggleAutoDetection() {
    autoDetectionEnabled.toggle()
    detectionTask?.cancel()
    if autoDetectionEnabled {
      manualMode = false
      showManualGuidance = false
      if image != nil {
        regions = []
        selectedRegionID = nil
        matches = []
        selectedItem = nil
        startDetection()
      } else {
        status = "上传截图以识别物品"
      }
    } else {
      stage = .idle
      progress = 0
      manualMode = image != nil
      showManualGuidance = image != nil
      status = image == nil ? "自动检测已关闭" : "自动检测已关闭，请手动选取"
    }
  }

  private func startDetection() {
    stage = .detecting
    progress = 0.08
    status = "正在查找道具"
    detectionTask = Task { [weak self] in
      await self?.runDetection()
    }
  }

  func dismissDetection() {
    stage = .idle
  }

  func cancelDetection() {
    detectionTask?.cancel()
    stage = .idle
    progress = 0
    status = image == nil ? "上传截图以识别物品" : "已取消自动检测，可手动选取"
  }

  func clearImage() {
    detectionTask?.cancel()
    selectionTask?.cancel()
    image = nil
    regions = []
    selectedRegionID = nil
    matches = []
    selectedItem = nil
    stage = .idle
    progress = 0
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
    status = manualMode ? "点击图片中物品进行手动选取" : "选择图中的检测框查看候选"
  }

  func startManualSelection() {
    stage = .idle
    manualMode = true
    showManualGuidance = true
    status = "点击图片中物品进行手动选取"
    Task { [weak self] in
      try? await Task.sleep(for: .seconds(4))
      self?.showManualGuidance = false
    }
  }

  func toggleManualSelection() {
    manualMode.toggle()
    showManualGuidance = manualMode
    status = manualMode ? "点击图片中物品进行手动选取" : "手动选取已关闭"
  }

  func addManualRegion(at point: CGPoint) {
    guard manualMode, let image else { return }
    let pixelWidth = image.size.width * image.scale
    let pixelHeight = image.size.height * image.scale
    let side = min(pixelWidth, pixelHeight) * manualBoxFraction
    let width = min(1, side / pixelWidth)
    let height = min(1, side / pixelHeight)
    let rect = CGRect(
      x: min(max(0, point.x - width / 2), 1 - width),
      y: min(max(0, point.y - height / 2), 1 - height),
      width: width,
      height: height
    )
    let region = DetectionRegion(rect: rect, automatic: false)
    regions.append(region)
    showManualGuidance = false
    select(region)
    status = "已添加手动选取"
  }

  func select(_ region: DetectionRegion) {
    selectedRegionID = region.id
    selectedItem = nil
    matches = []
    search(region)
  }

  func remove(_ region: DetectionRegion) {
    regions.removeAll { $0.id == region.id }
    if selectedRegionID == region.id {
      selectedRegionID = regions.first?.id
      matches = []
      selectedItem = nil
      if let first = regions.first { search(first) }
    }
    status = regions.isEmpty ? "未选中区域，可手动选取" : "已删除选中区域"
  }

  func thumbnail(for region: DetectionRegion) -> UIImage? {
    guard let image, let source = try? ImageUtilities.normalizedCGImage(image),
          let crop = try? ImageUtilities.squareCrop(source, normalizedRect: region.rect) else { return nil }
    return UIImage(cgImage: crop)
  }

  private func runDetection() async {
    guard let detector, let searchEngine, let image else {
      stage = .failed(status)
      return
    }
    do {
      let candidates = try await detector.detect(in: image)
      logger.info("detector candidates=\(candidates.count)")
      guard !Task.isCancelled else { return }
      progress = 0.62
      stage = .verifying
      status = "正在核对检测结果"
      var verified: [DetectionRegion] = []
      for (index, region) in candidates.enumerated() {
        let results = try await searchEngine.search(image: image, region: region, topK: 2)
        let best = results.first?.score ?? 0
        let second = results.dropFirst().first?.score ?? 0
        logger.info("candidate \(index) detector=\(region.detectorScore) best=\(best) second=\(second)")
        if best >= 0.76 || (best >= 0.70 && best - second >= 0.10 && region.detectorScore >= 0.70) {
          verified.append(region)
        }
        progress = 0.62 + 0.34 * Double(index + 1) / Double(max(1, candidates.count))
      }
      guard !Task.isCancelled else { return }
      regions = verified
      selectedRegionID = verified.first?.id
      progress = 1
      stage = .complete(verified.count)
      status = verified.isEmpty ? "未自动检测到道具，请手动选取" : "找到 \(verified.count) 个可能的道具"
      if let first = verified.first { search(first) }
    } catch {
      guard !Task.isCancelled else { return }
      progress = 1
      stage = .failed(error.localizedDescription)
      status = "自动检测未完成，请手动选取"
    }
  }

  private func search(_ region: DetectionRegion) {
    selectionTask?.cancel()
    guard let searchEngine, let image else { return }
    isSearching = true
    selectionTask = Task { [weak self] in
      do {
        let results = try await searchEngine.search(image: image, region: region, topK: 8)
        guard !Task.isCancelled, self?.selectedRegionID == region.id else { return }
        self?.matches = results
        self?.isSearching = false
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
