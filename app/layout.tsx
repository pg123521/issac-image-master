import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Isaac Item Lens",
  description: "完全在浏览器本地运行的截图物品识别工具。",
  manifest: "/manifest.webmanifest",
  icons: {
    icon: "/icon-192.png",
    apple: "/icon-192.png",
  },
  appleWebApp: {
    capable: true,
    statusBarStyle: "black-translucent",
    title: "Item Lens",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="zh-CN">
      <body>{children}</body>
    </html>
  );
}
