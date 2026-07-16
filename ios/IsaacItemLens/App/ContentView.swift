import PhotosUI
import SwiftUI

struct ContentView: View {
  @StateObject private var viewModel = IsaacLensViewModel()
  @State private var photoItem: PhotosPickerItem?
  @State private var showBoxSizeControl = false

  var body: some View {
    NavigationStack {
      ZStack {
        Color(uiColor: .systemBackground).ignoresSafeArea()
        if let image = viewModel.image {
          VStack(spacing: 0) {
            ScreenshotCanvas(
              image: image,
              regions: viewModel.regions,
              selectedRegionID: viewModel.selectedRegionID,
              manualMode: viewModel.manualMode,
              boxSizePreview: showBoxSizeControl ? CGFloat(viewModel.manualBoxFraction) : nil,
              onSelect: viewModel.select,
              onDelete: viewModel.remove,
              onManualPoint: viewModel.addManualRegion
            )
            .padding(12)
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            statusBar
          }
        } else {
          emptyState
        }

        if viewModel.showManualGuidance {
          VStack {
            Label("点击图片中物品进行手动选取", systemImage: "hand.tap")
              .font(.subheadline.bold())
              .padding(.horizontal, 16)
              .padding(.vertical, 11)
              .background(.regularMaterial, in: Capsule())
              .padding(.top, 12)
            Spacer()
          }
          .transition(.move(edge: .top).combined(with: .opacity))
          .allowsHitTesting(false)
        }

        if viewModel.stage.isPresented {
          DetectionDialog(
            stage: viewModel.stage,
            progress: viewModel.progress,
            onClose: viewModel.cancelDetection,
            onReview: viewModel.dismissDetection,
            onManual: viewModel.startManualSelection
          )
          .transition(.opacity)
        }
      }
      .animation(.easeInOut(duration: 0.2), value: viewModel.stage)
      .navigationTitle("Isaac Item Lens")
      .navigationBarTitleDisplayMode(.inline)
      .toolbar {
        if viewModel.image != nil {
          ToolbarItem(placement: .topBarLeading) {
            Button(action: viewModel.clearImage) {
              Image(systemName: "xmark")
            }
            .accessibilityLabel("退出当前截图")
          }
        }
        toolbar(hasImage: viewModel.image != nil)
      }
      .safeAreaInset(edge: .bottom, spacing: 0) {
        if viewModel.selectedRegion != nil && !viewModel.stage.isPresented {
          CandidatePanel(viewModel: viewModel, onClose: viewModel.closeSelection)
        }
      }
      .sheet(item: $viewModel.selectedItem) { item in
        ItemDetailView(item: item, repository: viewModel.repository) {
          viewModel.selectedItem = nil
        }
      }
      .onChange(of: photoItem) { _, item in
        showBoxSizeControl = false
        Task { await viewModel.importPhoto(item) }
      }
      .onChange(of: viewModel.manualMode) { _, enabled in
        if !enabled { showBoxSizeControl = false }
      }
    }
  }

  @ToolbarContentBuilder private func toolbar(hasImage: Bool) -> some ToolbarContent {
    ToolbarItemGroup(placement: .topBarTrailing) {
      Button(action: viewModel.toggleAutoDetection) {
        ZStack {
          Image(systemName: "viewfinder")
          if !viewModel.autoDetectionEnabled {
            Image(systemName: "slash.circle.fill")
              .font(.system(size: 10, weight: .bold))
              .symbolRenderingMode(.palette)
              .foregroundStyle(.white, .red)
              .offset(x: 7, y: 7)
          }
        }
      }
      .tint(viewModel.autoDetectionEnabled ? .primary : .red)
      .accessibilityLabel(viewModel.autoDetectionEnabled ? "关闭自动检测" : "开启自动检测")

      if viewModel.image != nil {
        Button {
          showBoxSizeControl.toggle()
        } label: {
          Image(systemName: "square.resize")
        }
        .disabled(!viewModel.manualMode)
        .accessibilityLabel("检测框大小")
        .popover(isPresented: $showBoxSizeControl, arrowEdge: .top) {
          VStack(alignment: .leading, spacing: 14) {
            Text("检测框大小")
              .font(.headline)
            HStack(spacing: 12) {
              Image(systemName: "square")
                .font(.caption)
                .foregroundStyle(.secondary)
              Slider(value: $viewModel.manualBoxFraction, in: 0.035...0.18)
                .tint(.yellow)
              Image(systemName: "square")
                .font(.title3)
                .foregroundStyle(.secondary)
            }
          }
          .padding(18)
          .frame(width: 290)
          .presentationCompactAdaptation(.popover)
        }

        Button(action: viewModel.toggleManualSelection) {
          Image(systemName: viewModel.manualMode ? "cursorarrow.click.2" : "cursorarrow.click")
        }
        .tint(viewModel.manualMode ? .yellow : .primary)
        .accessibilityLabel("手动选取")
      }

      PhotosPicker(selection: $photoItem, matching: .images) {
        Image(systemName: hasImage ? "arrow.triangle.2.circlepath" : "photo.badge.plus")
      }
      .accessibilityLabel(hasImage ? "换一张图" : "上传截图")
    }
  }

  private var emptyState: some View {
    VStack(spacing: 18) {
      Image(systemName: "viewfinder")
        .font(.system(size: 50, weight: .light))
        .foregroundStyle(.yellow)
      PhotosPicker(selection: $photoItem, matching: .images) {
        Text("上传截图以识别物品")
          .font(.title3.bold())
          .padding(.horizontal, 22)
          .padding(.vertical, 12)
      }
      .buttonStyle(.borderedProminent)
      .tint(.yellow)
      .foregroundStyle(.black)
    }
    .padding(28)
  }

  private var statusBar: some View {
    HStack(spacing: 8) {
      Circle()
        .fill(viewModel.manualMode ? Color.yellow : Color.green)
        .frame(width: 7, height: 7)
      Text(viewModel.status)
        .font(.caption)
        .lineLimit(1)
      Spacer()
      Label("离线", systemImage: "iphone.and.arrow.forward")
        .font(.caption)
        .foregroundStyle(.secondary)
    }
    .padding(.horizontal, 14)
    .padding(.vertical, 8)
    .background(Color(uiColor: .secondarySystemBackground))
  }
}
