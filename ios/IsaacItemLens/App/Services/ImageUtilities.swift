import CoreGraphics
import CoreVideo
import UIKit

enum ImageUtilities {
  static func normalizedCGImage(_ image: UIImage) throws -> CGImage {
    let size = image.size
    let format = UIGraphicsImageRendererFormat()
    format.scale = 1
    format.opaque = true
    let rendered = UIGraphicsImageRenderer(size: size, format: format).image { _ in
      image.draw(in: CGRect(origin: .zero, size: size))
    }
    guard let cgImage = rendered.cgImage else { throw ImageError.renderFailed }
    return cgImage
  }

  static func crop(_ image: CGImage, normalizedRect: CGRect) throws -> CGImage {
    let bounds = CGRect(x: 0, y: 0, width: image.width, height: image.height)
    let rect = CGRect(
      x: normalizedRect.minX * bounds.width,
      y: normalizedRect.minY * bounds.height,
      width: normalizedRect.width * bounds.width,
      height: normalizedRect.height * bounds.height
    ).integral.intersection(bounds)
    guard !rect.isEmpty, let result = image.cropping(to: rect) else { throw ImageError.cropFailed }
    return result
  }

  static func squareCrop(_ image: CGImage, normalizedRect: CGRect) throws -> CGImage {
    try crop(
      image,
      normalizedRect: squareNormalizedRect(
        imageWidth: CGFloat(image.width),
        imageHeight: CGFloat(image.height),
        normalizedRect: normalizedRect
      )
    )
  }

  static func squareNormalizedRect(
    imageWidth: CGFloat,
    imageHeight: CGFloat,
    normalizedRect: CGRect
  ) -> CGRect {
    let center = CGPoint(
      x: normalizedRect.midX * imageWidth,
      y: normalizedRect.midY * imageHeight
    )
    let side = min(
      min(imageWidth, imageHeight),
      max(normalizedRect.width * imageWidth, normalizedRect.height * imageHeight)
    )
    let x = min(max(0, center.x - side / 2), max(0, imageWidth - side))
    let y = min(max(0, center.y - side / 2), max(0, imageHeight - side))
    return CGRect(
      x: x / imageWidth,
      y: y / imageHeight,
      width: side / imageWidth,
      height: side / imageHeight
    )
  }

  static func pixelBuffer(
    from image: CGImage,
    width: Int,
    height: Int,
    letterbox: Bool,
    background: UIColor = UIColor(white: 114 / 255, alpha: 1)
  ) throws -> (CVPixelBuffer, ImageTransform) {
    var buffer: CVPixelBuffer?
    let attributes: [CFString: Any] = [
      kCVPixelBufferCGImageCompatibilityKey: true,
      kCVPixelBufferCGBitmapContextCompatibilityKey: true,
      kCVPixelBufferIOSurfacePropertiesKey: [:]
    ]
    let status = CVPixelBufferCreate(
      kCFAllocatorDefault,
      width,
      height,
      kCVPixelFormatType_32BGRA,
      attributes as CFDictionary,
      &buffer
    )
    guard status == kCVReturnSuccess, let buffer else { throw ImageError.pixelBufferFailed(status) }

    let sourceSize = CGSize(width: image.width, height: image.height)
    let targetSize = CGSize(width: width, height: height)
    let scale = letterbox
      ? min(targetSize.width / sourceSize.width, targetSize.height / sourceSize.height)
      : max(targetSize.width / sourceSize.width, targetSize.height / sourceSize.height)
    let drawSize = CGSize(width: sourceSize.width * scale, height: sourceSize.height * scale)
    let drawRect = CGRect(
      x: (targetSize.width - drawSize.width) / 2,
      y: (targetSize.height - drawSize.height) / 2,
      width: drawSize.width,
      height: drawSize.height
    )

    CVPixelBufferLockBaseAddress(buffer, [])
    defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
    guard let context = CGContext(
      data: CVPixelBufferGetBaseAddress(buffer),
      width: width,
      height: height,
      bitsPerComponent: 8,
      bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
      space: CGColorSpaceCreateDeviceRGB(),
      bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue
    ) else { throw ImageError.renderFailed }
    context.setFillColor(background.cgColor)
    context.fill(CGRect(origin: .zero, size: targetSize))
    context.interpolationQuality = .high
    context.translateBy(x: 0, y: targetSize.height)
    context.scaleBy(x: 1, y: -1)
    let flippedRect = CGRect(x: drawRect.minX, y: targetSize.height - drawRect.maxY, width: drawRect.width, height: drawRect.height)
    context.draw(image, in: flippedRect)
    return (buffer, ImageTransform(scale: scale, paddingX: drawRect.minX, paddingY: drawRect.minY))
  }
}

struct ImageTransform {
  let scale: CGFloat
  let paddingX: CGFloat
  let paddingY: CGFloat
}

enum ImageError: LocalizedError {
  case renderFailed
  case cropFailed
  case pixelBufferFailed(CVReturn)

  var errorDescription: String? {
    switch self {
    case .renderFailed: "无法处理图片"
    case .cropFailed: "选中区域无效"
    case .pixelBufferFailed(let status): "无法创建模型输入（\(status)）"
    }
  }
}
