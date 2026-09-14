/** @type {import('tailwindcss').Config} */
export default {
  content: ["./src/**/*.{astro,html,js,jsx,md,mdx,svelte,ts,tsx,vue}"],
  darkMode: "class",
  theme: {
    extend: {
      fontFamily: {
        sans: [
          "Inter",
          "-apple-system",
          "BlinkMacSystemFont",
          "SF Pro Display",
          "SF Pro Text",
          "Helvetica Neue",
          "Arial",
          "sans-serif",
        ],
        display: [
          "Inter Display",
          "Inter",
          "-apple-system",
          "SF Pro Display",
          "Helvetica Neue",
          "sans-serif",
        ],
        mono: [
          "JetBrains Mono",
          "SF Mono",
          "Menlo",
          "Monaco",
          "Consolas",
          "monospace",
        ],
      },
      colors: {
        ink: {
          50: "#f8f8f7",
          100: "#f1f1ef",
          200: "#e7e7e3",
          300: "#d3d3cd",
          400: "#a3a39a",
          500: "#6f6f66",
          600: "#4d4d46",
          700: "#34342f",
          800: "#1d1d1a",
          900: "#111110",
          950: "#0a0a09",
        },
        accent: {
          50: "#eef2ff",
          100: "#e0e7ff",
          200: "#c7d2fe",
          300: "#a5b4fc",
          400: "#818cf8",
          500: "#6366f1",
          600: "#4f46e5",
          700: "#4338ca",
          800: "#3730a3",
          900: "#312e81",
        },
        signal: {
          green: "#10b981",
          red: "#ef4444",
          amber: "#f59e0b",
          violet: "#8b5cf6",
          cyan: "#06b6d4",
        },
      },
      fontSize: {
        "display-2xl": ["clamp(3.5rem, 8vw, 6rem)", { lineHeight: "0.95", letterSpacing: "-0.04em", fontWeight: "600" }],
        "display-xl": ["clamp(2.75rem, 6vw, 4.5rem)", { lineHeight: "1.0", letterSpacing: "-0.035em", fontWeight: "600" }],
        "display-lg": ["clamp(2rem, 4.5vw, 3.25rem)", { lineHeight: "1.05", letterSpacing: "-0.03em", fontWeight: "600" }],
        "display-md": ["clamp(1.5rem, 3vw, 2.25rem)", { lineHeight: "1.15", letterSpacing: "-0.025em", fontWeight: "600" }],
        "display-sm": ["clamp(1.25rem, 2vw, 1.625rem)", { lineHeight: "1.25", letterSpacing: "-0.02em", fontWeight: "600" }],
        "eyebrow": ["0.75rem", { lineHeight: "1.4", letterSpacing: "0.18em", fontWeight: "500" }],
      },
      letterSpacing: {
        tightest: "-0.045em",
      },
      animation: {
        "fade-in": "fadeIn 0.8s ease-out forwards",
        "fade-up": "fadeUp 0.8s cubic-bezier(0.16, 1, 0.3, 1) forwards",
        "fade-up-slow": "fadeUp 1.2s cubic-bezier(0.16, 1, 0.3, 1) forwards",
        "draw-line": "drawLine 2s ease-out forwards",
        "pulse-soft": "pulseSoft 4s ease-in-out infinite",
        "shimmer": "shimmer 6s linear infinite",
        "float-slow": "float 8s ease-in-out infinite",
        "ticker": "ticker 40s linear infinite",
        "marquee": "marquee 50s linear infinite",
        "spin-slow": "spin 30s linear infinite",
      },
      keyframes: {
        fadeIn: {
          "0%": { opacity: "0" },
          "100%": { opacity: "1" },
        },
        fadeUp: {
          "0%": { opacity: "0", transform: "translateY(28px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        drawLine: {
          "0%": { strokeDashoffset: "1000" },
          "100%": { strokeDashoffset: "0" },
        },
        pulseSoft: {
          "0%, 100%": { opacity: "0.5" },
          "50%": { opacity: "1" },
        },
        shimmer: {
          "0%": { backgroundPosition: "-200% 0" },
          "100%": { backgroundPosition: "200% 0" },
        },
        float: {
          "0%, 100%": { transform: "translateY(0px)" },
          "50%": { transform: "translateY(-10px)" },
        },
        ticker: {
          "0%": { transform: "translateX(0)" },
          "100%": { transform: "translateX(-50%)" },
        },
        marquee: {
          "0%": { transform: "translateX(0)" },
          "100%": { transform: "translateX(-50%)" },
        },
      },
      backgroundImage: {
        "gradient-radial": "radial-gradient(var(--tw-gradient-stops))",
        "gradient-conic": "conic-gradient(from 180deg at 50% 50%, var(--tw-gradient-stops))",
        "gradient-brand":
          "linear-gradient(135deg, #6366f1 0%, #06b6d4 50%, #8b5cf6 100%)",
        "gradient-brand-soft":
          "linear-gradient(135deg, rgba(99,102,241,0.18) 0%, rgba(6,182,212,0.12) 50%, rgba(139,92,246,0.18) 100%)",
        "noise":
          "url(\"data:image/svg+xml;utf8,<svg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'><filter id='n'><feTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='2'/><feColorMatrix values='0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.06 0'/></filter><rect width='100%25' height='100%25' filter='url(%23n)'/></svg>\")",
      },
      boxShadow: {
        "soft-xl": "0 24px 60px -20px rgba(15, 23, 42, 0.18), 0 8px 18px -8px rgba(15, 23, 42, 0.08)",
        "soft-lg": "0 16px 40px -16px rgba(15, 23, 42, 0.14), 0 6px 14px -6px rgba(15, 23, 42, 0.06)",
        "glow-brand": "0 0 0 1px rgba(99, 102, 241, 0.16), 0 12px 40px -12px rgba(99, 102, 241, 0.4)",
      },
    },
  },
  plugins: [],
};
