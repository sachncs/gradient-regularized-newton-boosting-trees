# GRNBT Site

Premium product page for [GRNBT](https://github.com/sachncs/gradient-regularized-newton-boosting-trees) — built with [Astro 5](https://astro.build/) and [Tailwind CSS](https://tailwindcss.com/).

## Stack

- **Astro 5** — static-first, content-driven, minimal JS
- **Tailwind CSS 3** — utility styling with custom design tokens
- **TypeScript** — strict mode
- **Inter + JetBrains Mono** — typography

## Develop

```bash
cd site
npm install
npm run dev          # http://localhost:4321
```

## Build

```bash
npm run build        # → site/dist/
npm run preview      # preview the production build locally
```

## Structure

```
site/
├── public/                  # Static assets (favicon, robots.txt)
├── src/
│   ├── components/          # Page sections
│   ├── layouts/             # HTML shell, SEO, theme bootstrap
│   ├── pages/               # Routes (index, 404)
│   └── styles/global.css    # Design tokens & utilities
├── astro.config.mjs
├── tailwind.config.mjs
└── tsconfig.json
```

## Deployment

Configured for **GitHub Pages** via the workflow at
`../.github/workflows/site.yml`. The base path is
`/gradient-regularized-newton-boosting-trees` — Astro's
[`base`](https://docs.astro.build/en/reference/configuration-reference/#base)
flag handles asset prefixing automatically.

## Design tokens

All visual tokens (colors, typography, animations) live in
`tailwind.config.mjs` and `src/styles/global.css`. The page supports
**dark / light** themes with a `localStorage` toggle and
`prefers-color-scheme` default.
