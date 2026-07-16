import Foundation
import SwiftUI

struct CandidatePanel: View {
  @ObservedObject var viewModel: IsaacLensViewModel
  let onClose: () -> Void
  @State private var showCandidateLimitPicker = false

  var body: some View {
    VStack(alignment: .leading, spacing: 9) {
      HStack {
        if let region = viewModel.selectedRegion, let thumbnail = viewModel.thumbnail(for: region) {
          Image(uiImage: thumbnail)
            .resizable()
            .scaledToFill()
            .frame(width: 38, height: 38)
            .clipShape(RoundedRectangle(cornerRadius: 4))
            .overlay { RoundedRectangle(cornerRadius: 4).stroke(Color.yellow, lineWidth: 1.5) }
        }
        VStack(alignment: .leading, spacing: 2) {
          Text("选中区域")
            .font(.subheadline.bold())
          Text("Top \(viewModel.candidateDisplayLimit) 相似对象")
            .font(.caption)
            .foregroundStyle(.secondary)
        }
        Spacer()
        if viewModel.isSearching {
          ProgressView().controlSize(.small)
        }
        Button {
          showCandidateLimitPicker.toggle()
        } label: {
          Label("\(viewModel.candidateDisplayLimit)", systemImage: "list.number")
            .font(.caption.bold())
        }
        .buttonStyle(.bordered)
        .controlSize(.small)
        .accessibilityLabel("选择候选数量")
        .popover(isPresented: $showCandidateLimitPicker, arrowEdge: .bottom) {
          VStack(alignment: .leading, spacing: 12) {
            Text("候选数量")
              .font(.headline)
            Stepper(
              value: $viewModel.candidateDisplayLimit,
              in: 1...50,
              step: 1
            ) {
              Text("显示 Top \(viewModel.candidateDisplayLimit)")
                .monospacedDigit()
            }
          }
          .padding(18)
          .frame(minWidth: 220)
          .presentationCompactAdaptation(.popover)
        }
        Button(action: onClose) {
          Image(systemName: "xmark.circle.fill")
            .font(.title3)
            .foregroundStyle(.secondary)
        }
        .accessibilityLabel("收起候选道具")
      }
      if viewModel.matches.isEmpty && !viewModel.isSearching {
        Text("没有找到相似对象")
          .font(.subheadline)
          .foregroundStyle(.secondary)
          .frame(maxWidth: .infinity, minHeight: 78)
      } else {
        ScrollView(.horizontal, showsIndicators: false) {
          LazyHStack(spacing: 10) {
            ForEach(viewModel.matches.prefix(viewModel.candidateDisplayLimit)) { match in
              Button {
                viewModel.selectedItem = match.item
              } label: {
                VStack(spacing: 3) {
                  ObjectIcon(item: match.item, repository: viewModel.repository)
                    .frame(width: 48, height: 48)
                  Text(String(format: "%.1f%%", match.score * 100))
                    .font(.caption2.bold().monospacedDigit())
                    .foregroundStyle(.orange)
                }
                .padding(5)
                .frame(width: 64, height: 70)
                .background(Color(uiColor: .tertiarySystemBackground), in: RoundedRectangle(cornerRadius: 6))
                .overlay {
                  RoundedRectangle(cornerRadius: 6).stroke(Color.white.opacity(0.08))
                }
              }
              .buttonStyle(.plain)
              .accessibilityLabel("\(match.item.nameZh)，相似度 \(String(format: "%.1f%%", match.score * 100))")
            }
          }
        }
      }
    }
    .padding(.horizontal, 16)
    .padding(.top, 9)
    .padding(.bottom, 8)
    .frame(height: 136, alignment: .top)
    .background(.thinMaterial)
    .overlay(alignment: .top) { Divider() }
  }
}

struct ObjectIcon: View {
  let item: IsaacObject
  let repository: ItemRepository?

  var body: some View {
    Group {
      if let icon = repository?.icon(for: item) {
        Image(uiImage: icon)
          .resizable()
          .interpolation(.none)
          .scaledToFit()
      } else {
        Image(systemName: "questionmark.square.dashed")
          .resizable()
          .scaledToFit()
          .foregroundStyle(.secondary)
      }
    }
    .padding(5)
    .background(Color.black.opacity(0.45), in: RoundedRectangle(cornerRadius: 5))
  }
}
