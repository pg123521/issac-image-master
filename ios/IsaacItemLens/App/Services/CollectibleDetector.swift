import CoreGraphics
import CoreML
import OSLog
import UIKit

actor CollectibleDetector {
  private let logger = Logger(subsystem: "com.pg123521.IsaacItemLens", category: "detector")
  private let model: MLModel
  private let inputSize = 1024
  private let confidenceThreshold: Float = 0.25

  init() throws {
    model = try CoreMLModelLoader.load("RoomCollectibleDetector")
  }

  func detect(in image: UIImage) throws -> [DetectionRegion] {
    let source = try ImageUtilities.normalizedCGImage(image)
    let imageBounds = CGRect(x: 0, y: 0, width: source.width, height: source.height)
    var detections: [ScoredRect] = []
    for tileRect in detectionTiles(width: source.width, height: source.height) {
      guard let tile = source.cropping(to: tileRect.integral) else { continue }
      let (buffer, transform) = try ImageUtilities.pixelBuffer(
        from: tile,
        width: inputSize,
        height: inputSize,
        letterbox: true
      )
      let provider = try MLDictionaryFeatureProvider(dictionary: ["image": MLFeatureValue(pixelBuffer: buffer)])
      let output = try model.prediction(from: provider)
      guard let array = output.featureValue(for: "var_1440")?.multiArrayValue else {
        throw ModelError.missingOutput("var_1440")
      }
      detections.append(contentsOf: try parse(array, tileRect: tileRect, transform: transform, imageBounds: imageBounds))
    }
    let suppressed = nonMaximumSuppression(detections, threshold: 0.42)
      .prefix(20)
      .map { scored in
        let normalized = CGRect(
          x: scored.rect.minX / imageBounds.width,
          y: scored.rect.minY / imageBounds.height,
          width: scored.rect.width / imageBounds.width,
          height: scored.rect.height / imageBounds.height
        )
        return DetectionRegion(rect: normalized, detectorScore: scored.score, automatic: true)
      }
    let roomRegions = suppressed.filter(isInsideRoomFloor)
    logger.info("raw detections=\(detections.count) nms=\(suppressed.count) room=\(roomRegions.count)")
    return roomRegions
  }

  private func parse(
    _ array: MLMultiArray,
    tileRect: CGRect,
    transform: ImageTransform,
    imageBounds: CGRect
  ) throws -> [ScoredRect] {
    guard array.shape.count == 3, array.shape[2].intValue == 6 else {
      throw ModelError.invalidOutput("var_1440")
    }
    let rows = array.shape[1].intValue
    let rowStride = array.strides[1].intValue
    let fieldStride = array.strides[2].intValue
    var results: [ScoredRect] = []
    for row in 0..<rows {
      let score = array[row * rowStride + 4 * fieldStride].floatValue
      guard score >= confidenceThreshold else { continue }
      let x1 = CGFloat(array[row * rowStride].floatValue)
      let y1 = CGFloat(array[row * rowStride + fieldStride].floatValue)
      let x2 = CGFloat(array[row * rowStride + 2 * fieldStride].floatValue)
      let y2 = CGFloat(array[row * rowStride + 3 * fieldStride].floatValue)
      let rect = CGRect(
        x: tileRect.minX + (x1 - transform.paddingX) / transform.scale,
        y: tileRect.minY + (y1 - transform.paddingY) / transform.scale,
        width: (x2 - x1) / transform.scale,
        height: (y2 - y1) / transform.scale
      ).standardized.intersection(imageBounds)
      if !rect.isEmpty { results.append(ScoredRect(rect: rect, score: score)) }
    }
    return results
  }

  private func detectionTiles(width: Int, height: Int) -> [CGRect] {
    let full = CGRect(x: 0, y: 0, width: width, height: height)
    guard CGFloat(width) > CGFloat(height) * 1.35 else { return [full] }
    let side = CGFloat(height)
    let travel = CGFloat(width) - side
    let count = max(2, min(4, Int((CGFloat(width) / CGFloat(height)).rounded()) + 1))
    var tiles = [full]
    for index in 0..<count {
      let left = travel * CGFloat(index) / CGFloat(max(1, count - 1))
      tiles.append(CGRect(x: left, y: 0, width: side, height: side))
    }
    return tiles
  }

  private func nonMaximumSuppression(_ boxes: [ScoredRect], threshold: CGFloat) -> [ScoredRect] {
    var kept: [ScoredRect] = []
    for candidate in boxes.sorted(by: { $0.score > $1.score }) {
      if kept.allSatisfy({ intersectionOverUnion(candidate.rect, $0.rect) < threshold }) {
        kept.append(candidate)
      }
    }
    return kept
  }

  private func intersectionOverUnion(_ lhs: CGRect, _ rhs: CGRect) -> CGFloat {
    let intersection = lhs.intersection(rhs)
    let area = intersection.isNull ? 0 : intersection.width * intersection.height
    return area / max(0.0001, lhs.width * lhs.height + rhs.width * rhs.height - area)
  }

  private func isInsideRoomFloor(_ region: DetectionRegion) -> Bool {
    let center = CGPoint(x: region.rect.midX, y: region.rect.midY)
    return center.x >= 0.12 && center.x <= 0.88 && center.y >= 0.24 && center.y <= 0.82
  }
}

private struct ScoredRect {
  let rect: CGRect
  let score: Float
}
