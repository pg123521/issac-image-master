import Accelerate
import CoreML
import OSLog
import UIKit

actor ItemSearchEngine {
  private let logger = Logger(subsystem: "com.pg123521.IsaacItemLens", category: "search")
  private let model: MLModel
  private let repository: ItemRepository

  init(repository: ItemRepository) throws {
    self.repository = repository
    // The iOS Simulator's MPS backend returns an all-zero FP16 embedding for
    // this converted encoder. CPU execution is deterministic on simulator and device.
    model = try CoreMLModelLoader.load("MobileCLIPImageEncoderRaw", computeUnits: .cpuOnly)
  }

  func search(image: UIImage, region: DetectionRegion, topK: Int) throws -> [SearchMatch] {
    let requestID = String(UUID().uuidString.prefix(8))
    let source = try ImageUtilities.normalizedCGImage(image)
    let crop = try ImageUtilities.squareCrop(source, normalizedRect: region.rect)
    let (buffer, _) = try ImageUtilities.pixelBuffer(from: crop, width: 256, height: 256, letterbox: false, background: .black)
    let input = pixelSummary(buffer)
    let regionText = rectDescription(region.rect)
    let inputText = "region=\(regionText) source=\(source.width)x\(source.height) crop=\(crop.width)x\(crop.height) input=\(input)"
    logger.info("search begin id=\(requestID, privacy: .public) \(inputText, privacy: .public)")
    let provider = try MLDictionaryFeatureProvider(dictionary: ["image": MLFeatureValue(pixelBuffer: buffer)])
    let output = try model.prediction(from: provider)
    guard let embedding = output.featureValue(for: "embedding")?.multiArrayValue else {
      logger.error("search failed id=\(requestID, privacy: .public) missing embedding output")
      throw ModelError.missingOutput("embedding")
    }
    let query = try readEmbedding(embedding, requestID: requestID)
    let matches = try topMatches(query: query, count: topK, requestID: requestID)
    let result = matches.prefix(3).map { "\($0.item.id):\(String(format: "%.4f", $0.score))" }.joined(separator: ",")
    logger.info("search complete id=\(requestID, privacy: .public) top=\(result, privacy: .public)")
    return matches
  }

  private func readEmbedding(_ embedding: MLMultiArray, requestID: String) throws -> [Float] {
    guard embedding.count == repository.dimensions else {
      logger.error("embedding invalid id=\(requestID, privacy: .public) count=\(embedding.count) expected=\(self.repository.dimensions)")
      throw ModelError.invalidOutput("embedding 维度为 \(embedding.count)，预期为 \(repository.dimensions)")
    }
    let values: [Float]
    switch embedding.dataType {
    case .float16:
      values = MLShapedArray<Float16>(embedding).scalars.map(Float.init)
    case .float32:
      values = MLShapedArray<Float>(embedding).scalars
    case .double:
      values = MLShapedArray<Double>(embedding).scalars.map(Float.init)
    default:
      values = (0..<embedding.count).map { embedding[$0].floatValue }
    }

    let finiteCount = values.reduce(into: 0) { count, value in
      if value.isFinite { count += 1 }
    }
    let finiteValues = values.filter(\.isFinite)
    let minimum = finiteValues.min() ?? .nan
    let maximum = finiteValues.max() ?? .nan
    let normSquared = values.reduce(Float.zero) { $0 + $1 * $1 }
    let shape = embedding.shape.map(\.stringValue).joined(separator: "x")
    let strides = embedding.strides.map(\.stringValue).joined(separator: ",")
    let embeddingText = "type=\(embedding.dataType) shape=\(shape) strides=\(strides) finite=\(finiteCount)/\(values.count) min=\(minimum) max=\(maximum) norm2=\(normSquared)"
    logger.info("embedding id=\(requestID, privacy: .public) \(embeddingText, privacy: .public)")
    guard normSquared.isFinite, normSquared > 1e-8, values.allSatisfy({ $0.isFinite }) else {
      logger.error("embedding rejected id=\(requestID, privacy: .public) finite=\(finiteCount)/\(values.count) norm2=\(normSquared)")
      let reason = normSquared == 0 ? "embedding 全为 0" : "embedding 包含无效值"
      throw ModelError.invalidOutput(reason)
    }
    let inverseNorm = 1 / normSquared.squareRoot()
    return values.map { $0 * inverseNorm }
  }

  private func topMatches(query: [Float], count: Int, requestID: String) throws -> [SearchMatch] {
    let dimensions = repository.dimensions
    let rows = repository.objects.count
    var scored = [(index: Int, score: Float)]()
    scored.reserveCapacity(rows)
    query.withUnsafeBufferPointer { queryBuffer in
      repository.vectors.withUnsafeBufferPointer { vectorsBuffer in
        guard let queryBase = queryBuffer.baseAddress, let vectorsBase = vectorsBuffer.baseAddress else { return }
        for row in 0..<rows {
          var score: Float = 0
          vDSP_dotpr(
            queryBase,
            1,
            vectorsBase.advanced(by: row * dimensions),
            1,
            &score,
            vDSP_Length(dimensions)
          )
          scored.append((row, score))
        }
      }
    }
    let minimum = scored.map({ $0.score }).min()
    let maximum = scored.map({ $0.score }).max()
    guard let minimum,
          let maximum,
          minimum.isFinite,
          maximum.isFinite,
          maximum - minimum > 1e-5 else {
      logger.error("scores rejected id=\(requestID, privacy: .public) min=\(minimum ?? .nan) max=\(maximum ?? .nan)")
      throw ModelError.invalidOutput("相似度分数没有区分度")
    }
    return scored
      .sorted(by: { $0.score > $1.score })
      .prefix(count)
      .map { SearchMatch(item: repository.objects[$0.index], score: $0.score) }
  }

  private func rectDescription(_ rect: CGRect) -> String {
    String(format: "%.4f,%.4f,%.4f,%.4f", rect.origin.x, rect.origin.y, rect.width, rect.height)
  }

  private func pixelSummary(_ buffer: CVPixelBuffer) -> String {
    CVPixelBufferLockBaseAddress(buffer, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
    guard let base = CVPixelBufferGetBaseAddress(buffer) else { return "unavailable" }
    let height = CVPixelBufferGetHeight(buffer)
    let bytesPerRow = CVPixelBufferGetBytesPerRow(buffer)
    let byteCount = height * bytesPerRow
    let bytes = base.assumingMemoryBound(to: UInt8.self)
    let stride = max(1, byteCount / 4096)
    var hash: UInt64 = 1469598103934665603
    var sum: UInt64 = 0
    var index = 0
    while index < byteCount {
      let value = bytes[index]
      hash = (hash ^ UInt64(value)) &* 1099511628211
      sum += UInt64(value)
      index += stride
    }
    return String(format: "hash=%016llx mean=%.2f", hash, Double(sum) / Double(max(1, (byteCount + stride - 1) / stride)))
  }
}
