import { afterEach, describe, expect, it, vi } from "vitest";

afterEach(() => {
  vi.unstubAllEnvs();
});

describe("dashboard paths", () => {
  it.each([
    {
      root: "",
      proxy: "",
      asset: "/litellm-asset-prefix",
      favicon: "/get_favicon",
      discovery: "/litellm/.well-known/litellm-ui-config",
      output: "export",
    },
    {
      root: " / ",
      proxy: "",
      asset: "/litellm-asset-prefix",
      favicon: "/get_favicon",
      discovery: "/litellm/.well-known/litellm-ui-config",
      output: "export",
    },
    {
      root: " /gateway/ ",
      proxy: "",
      asset: "/gateway",
      favicon: "/gateway/get_favicon",
      discovery: "/gateway/.well-known/litellm-ui-config",
      output: "export",
    },
    {
      root: "",
      proxy: "http://localhost:4000",
      asset: "",
      favicon: "/get_favicon",
      discovery: "/.well-known/litellm-ui-config",
      output: undefined,
    },
    {
      root: "/gateway/",
      proxy: "http://localhost:4000",
      asset: "/gateway",
      favicon: "/gateway/get_favicon",
      discovery: "/gateway/.well-known/litellm-ui-config",
      output: undefined,
    },
  ])(
    "configures paths for root $root and dev proxy $proxy",
    async ({ root, proxy, asset, favicon, discovery, output }) => {
      vi.stubEnv("SERVER_ROOT_PATH", root);
      vi.stubEnv("LENS_DEV_PROXY_URL", proxy);
      vi.resetModules();
      const { default: config } = await import("../next.config.mjs");

      expect(config.assetPrefix).toBe(asset);
      expect(config.env.NEXT_PUBLIC_LITELLM_FAVICON_PATH).toBe(favicon);
      expect(config.env.NEXT_PUBLIC_LITELLM_UI_CONFIG_PATH).toBe(discovery);
      expect(config.output).toBe(output);
    },
  );
});
