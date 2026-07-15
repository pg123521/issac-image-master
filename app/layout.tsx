import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Isaac Item Lens",
  description: "Offline screenshot item recognition prototype for The Binding of Isaac.",
  icons: {
    icon: "/favicon.svg",
    shortcut: "/favicon.svg",
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
