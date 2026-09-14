/**
 * Resolve a path relative to Astro's base URL.
 *
 * `import.meta.env.BASE_URL` may or may not include a trailing slash
 * depending on the Astro/Vite version. This helper normalizes it so
 * callers can use `withBase("/favicon.svg")` and get a correct URL
 * regardless of how the base was configured.
 */
export function withBase(path: string): string {
  const raw = import.meta.env.BASE_URL ?? "/";
  const base = raw.endsWith("/") ? raw : raw + "/";
  if (!path) return base;
  if (path.startsWith("/")) return base + path.slice(1);
  return base + path;
}
