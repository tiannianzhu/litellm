import path from "path";
import { fileURLToPath } from "url";

/** @type {import('next').NextConfig} */
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const normalizedPath = (process.env.SERVER_ROOT_PATH ?? "").trim().replace(/^\/+|\/+$/g, "");
const serverRootPath = normalizedPath === "" ? "" : `/${normalizedPath}`;

const nextConfig = {
  output: "export",
  compiler: {
    removeConsole: process.env.NODE_ENV === "production" ? { exclude: ["error", "warn"] } : false,
  },
  // Required with output: "export" — default image optimizer runs only in server mode.
  // See https://nextjs.org/docs/messages/export-image-api
  images: {
    unoptimized: true,
  },
  basePath: "",
  assetPrefix: serverRootPath,
  env: {
    NEXT_PUBLIC_LITELLM_FAVICON_PATH: `${serverRootPath}/get_favicon`,
    NEXT_PUBLIC_LITELLM_UI_CONFIG_PATH: `${serverRootPath}/.well-known/litellm-ui-config`,
  },
  trailingSlash: true,
  turbopack: {
    // Must be absolute; "." is no longer allowed
    root: __dirname,
  },
};

export default nextConfig;
