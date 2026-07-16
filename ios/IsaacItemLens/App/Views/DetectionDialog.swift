import SwiftUI

struct DetectionDialog: View {
  let stage: DetectionStage
  let progress: Double
  let onClose: () -> Void
  let onReview: () -> Void
  let onManual: () -> Void

  var body: some View {
    ZStack {
      Color.black.opacity(0.62).ignoresSafeArea()
      ZStack(alignment: .topTrailing) {
        VStack(spacing: 18) {
          statusMark
          Text(title)
            .font(.title3.bold())
            .multilineTextAlignment(.center)
          Text(message)
            .font(.subheadline)
            .foregroundStyle(.secondary)
            .multilineTextAlignment(.center)
          ProgressView(value: progress)
            .tint(.yellow)
          if isFinished {
            HStack(spacing: 12) {
              if hasAutomaticResults {
                Button("查看自动检测结果", action: onReview)
                  .buttonStyle(.bordered)
              }
              Button("手动选取", action: onManual)
                .buttonStyle(.borderedProminent)
                .tint(.yellow)
                .foregroundStyle(.black)
            }
          }
        }
        Button(action: onClose) {
          Image(systemName: "xmark.circle.fill")
            .font(.title2)
            .foregroundStyle(.secondary)
        }
        .accessibilityLabel("关闭检测窗口")
      }
      .padding(24)
      .frame(maxWidth: 390)
      .background(Color(uiColor: .secondarySystemBackground), in: RoundedRectangle(cornerRadius: 8))
      .padding(24)
    }
  }

  @ViewBuilder private var statusMark: some View {
    switch stage {
    case .complete(let count) where count > 0:
      Image(systemName: "checkmark.circle.fill").font(.system(size: 38)).foregroundStyle(.green)
    case .complete, .failed:
      Image(systemName: "hand.tap.fill").font(.system(size: 34)).foregroundStyle(.yellow)
    default:
      ProgressView().controlSize(.large).tint(.yellow)
    }
  }

  private var title: String {
    switch stage {
    case .detecting: "正在查找道具"
    case .verifying: "正在核对候选"
    case .complete(let count): count > 0 ? "找到 \(count) 个道具" : "未自动检测到道具，请手动选取"
    case .failed: "自动检测未完成"
    case .idle: ""
    }
  }

  private var message: String {
    switch stage {
    case .detecting: "正在扫描房间画面，请稍候。"
    case .verifying: "正在排除角色、界面与普通拾取物。"
    case .complete(let count): count > 0 ? "请检查检测结果，遗漏时可继续手动选取。" : "请点击图片中的道具进行选取。"
    case .failed(let reason): reason
    case .idle: ""
    }
  }

  private var isFinished: Bool {
    if case .complete = stage { return true }
    if case .failed = stage { return true }
    return false
  }

  private var hasAutomaticResults: Bool {
    if case .complete(let count) = stage { return count > 0 }
    return false
  }
}
