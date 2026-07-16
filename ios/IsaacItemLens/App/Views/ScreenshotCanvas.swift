import SwiftUI
import UIKit

struct ScreenshotCanvas: View {
  let image: UIImage
  let regions: [DetectionRegion]
  let selectedRegionID: UUID?
  let manualMode: Bool
  let boxSizePreview: CGFloat?
  let onSelect: (DetectionRegion) -> Void
  let onDelete: (DetectionRegion) -> Void
  let onManualPoint: (CGPoint) -> Void

  var body: some View {
    GeometryReader { proxy in
      let imageSize = CGSize(width: image.size.width * image.scale, height: image.size.height * image.scale)
      let frame = aspectFitFrame(content: imageSize, container: proxy.size)
      ZStack(alignment: .topLeading) {
        Color.black.opacity(0.45)
        Image(uiImage: image)
          .resizable()
          .interpolation(.high)
          .scaledToFit()
          .frame(width: frame.width, height: frame.height)
          .position(x: frame.midX, y: frame.midY)

        Color.clear
          .frame(width: frame.width, height: frame.height)
          .contentShape(Rectangle())
          .onTapGesture { location in
            guard manualMode else { return }
            onManualPoint(CGPoint(
              x: min(max(0, location.x / frame.width), 1),
              y: min(max(0, location.y / frame.height), 1)
            ))
          }
          .position(x: frame.midX, y: frame.midY)

        ForEach(regions) { region in
          let cropRect = ImageUtilities.squareNormalizedRect(
            imageWidth: imageSize.width,
            imageHeight: imageSize.height,
            normalizedRect: region.rect
          )
          let overlay = CGRect(
            x: frame.minX + cropRect.minX * frame.width,
            y: frame.minY + cropRect.minY * frame.height,
            width: cropRect.width * frame.width,
            height: cropRect.height * frame.height
          )
          RegionOverlay(
            selected: region.id == selectedRegionID,
            automatic: region.automatic,
            onSelect: { onSelect(region) },
            onDelete: { onDelete(region) }
          )
          .frame(width: max(1, overlay.width), height: max(1, overlay.height))
          .position(x: overlay.midX, y: overlay.midY)
        }

        if let boxSizePreview {
          let side = min(imageSize.width, imageSize.height) * boxSizePreview
          let previewWidth = side / imageSize.width * frame.width
          let previewHeight = side / imageSize.height * frame.height
          BoxSizePreview()
            .frame(width: previewWidth, height: previewHeight)
            .position(x: frame.midX, y: frame.midY)
            .transition(.scale(scale: 0.9).combined(with: .opacity))
            .allowsHitTesting(false)
        }
      }
      .animation(.easeOut(duration: 0.14), value: boxSizePreview)
    }
    .clipShape(RoundedRectangle(cornerRadius: 6))
  }

  private func aspectFitFrame(content: CGSize, container: CGSize) -> CGRect {
    let scale = min(container.width / content.width, container.height / content.height)
    let size = CGSize(width: content.width * scale, height: content.height * scale)
    return CGRect(
      x: (container.width - size.width) / 2,
      y: (container.height - size.height) / 2,
      width: size.width,
      height: size.height
    )
  }
}

private struct BoxSizePreview: View {
  var body: some View {
    RoundedRectangle(cornerRadius: 3)
      .fill(Color.yellow.opacity(0.12))
      .overlay {
        RoundedRectangle(cornerRadius: 3)
          .stroke(.black.opacity(0.7), lineWidth: 6)
      }
      .overlay {
        RoundedRectangle(cornerRadius: 3)
          .stroke(Color.yellow, style: StrokeStyle(lineWidth: 3, dash: [7, 5]))
      }
      .overlay {
        Image(systemName: "scope")
          .font(.system(size: 14, weight: .semibold))
          .foregroundStyle(.white)
          .shadow(color: .black, radius: 2)
      }
      .shadow(color: Color.yellow.opacity(0.75), radius: 8)
  }
}

private struct RegionOverlay: View {
  let selected: Bool
  let automatic: Bool
  let onSelect: () -> Void
  let onDelete: () -> Void

  var body: some View {
    ZStack(alignment: .topTrailing) {
      Button(action: onSelect) {
        RoundedRectangle(cornerRadius: 2)
          .fill((selected ? Color.yellow : Color.cyan).opacity(selected ? 0.16 : 0.07))
          .overlay {
            RoundedRectangle(cornerRadius: 2)
              .stroke(.black.opacity(selected ? 0.72 : 0.35), lineWidth: selected ? 6 : 4)
          }
          .overlay {
            RoundedRectangle(cornerRadius: 2)
              .stroke(selected ? Color.yellow : Color.cyan, lineWidth: selected ? 3 : 2)
          }
          .overlay {
            if selected {
              RoundedRectangle(cornerRadius: 1)
                .inset(by: 4)
                .stroke(.white.opacity(0.9), lineWidth: 1)
            }
          }
          .shadow(color: selected ? Color.yellow.opacity(0.9) : .clear, radius: 9)
          .shadow(color: selected ? Color.white.opacity(0.45) : .clear, radius: 3)
      }
      .buttonStyle(.plain)
      .accessibilityLabel(automatic ? "自动检测区域" : "手动选取区域")

      if selected {
        Image(systemName: "checkmark")
          .font(.system(size: 10, weight: .black))
          .foregroundStyle(.black)
          .frame(width: 20, height: 20)
          .background(Color.yellow, in: Circle())
          .overlay(Circle().stroke(.white, lineWidth: 1))
          .shadow(color: .black.opacity(0.55), radius: 3, y: 1)
          .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .bottomLeading)
          .offset(x: -8, y: 8)
          .allowsHitTesting(false)
      }

      Button(action: onDelete) {
        Image(systemName: "xmark")
          .font(.system(size: 11, weight: .bold))
          .frame(width: 24, height: 24)
          .foregroundStyle(.white)
          .background(Color.red, in: Circle())
      }
      .buttonStyle(.plain)
      .offset(x: 12, y: -12)
      .accessibilityLabel("删除区域")
    }
  }
}
