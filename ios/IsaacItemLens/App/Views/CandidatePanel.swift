import SwiftUI

struct CandidatePanel: View {
  @ObservedObject var viewModel: IsaacLensViewModel
  let onClose: () -> Void

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
          Text("相似道具")
            .font(.caption)
            .foregroundStyle(.secondary)
        }
        Spacer()
        if viewModel.isSearching {
          ProgressView().controlSize(.small)
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
            ForEach(viewModel.matches) { match in
              Button {
                viewModel.selectedItem = match.item
              } label: {
                HStack(spacing: 9) {
                  ObjectIcon(item: match.item, repository: viewModel.repository)
                    .frame(width: 48, height: 48)
                  VStack(alignment: .leading, spacing: 4) {
                    Text(match.item.nameZh)
                      .font(.subheadline.bold())
                      .foregroundStyle(.primary)
                      .lineLimit(1)
                    Text(match.item.nameEn)
                      .font(.caption)
                      .foregroundStyle(.secondary)
                      .lineLimit(1)
                  }
                }
                .padding(8)
                .frame(width: 166, height: 66, alignment: .leading)
                .background(Color(uiColor: .tertiarySystemBackground), in: RoundedRectangle(cornerRadius: 6))
                .overlay {
                  RoundedRectangle(cornerRadius: 6).stroke(Color.white.opacity(0.08))
                }
              }
              .buttonStyle(.plain)
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
