/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        background: "#0B1220",
        surface: "#111827",
        "surface-elevated": "#151E2D",
        border: "#1F2937",
        "text-primary": "#F9FAFB",
        "text-secondary": "#9CA3AF",
        positive: "#10B981",
        negative: "#EF4444",
        accent: "#3B82F6",
        "accent-hover": "#2563EB",
        warning: "#F59E0B",
      },
      fontFamily: {
        sans: ["Geist", "Inter", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "Fira Code", "monospace"],
        number: ["Geist", "Inter", "monospace"],
      },
      fontSize: {
        "metric-lg": ["2rem", { lineHeight: "2.5rem", fontWeight: "600" }],
        "metric-md": ["1.5rem", { lineHeight: "2rem", fontWeight: "600" }],
        "metric-sm": ["1.125rem", { lineHeight: "1.5rem", fontWeight: "500" }],
      },
      borderRadius: {
        panel: "0.5rem",
      },
    },
  },
  plugins: [require("tailwindcss-animate")],
};
