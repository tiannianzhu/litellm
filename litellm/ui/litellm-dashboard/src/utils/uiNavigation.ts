import { getUiPageUrl } from "@/components/networking";
import { uiHref } from "@/utils/uiHref";

export function navigateToUiPage(url: string, replace = false): void {
  const fallback = (() => {
    const candidate = uiHref("").replace(/\/$/, "") || "/";
    try {
      return new URL(candidate, window.location.origin).origin === window.location.origin ? candidate : "/ui";
    } catch {
      return "/ui";
    }
  })();
  const destination = (() => {
    try {
      const target = new URL(getUiPageUrl(url), window.location.origin);
      const isHttp = target.protocol === "http:" || target.protocol === "https:";

      return isHttp && target.origin === window.location.origin ? target.href : fallback;
    } catch {
      return fallback;
    }
  })();

  if (replace) {
    window.location.replace(destination);
  } else {
    window.location.assign(destination);
  }
}
