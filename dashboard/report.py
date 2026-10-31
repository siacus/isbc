"""Print a self-contained dashboard snapshot, without accessing live app data."""
import asyncio
import base64
from html import escape
from html.parser import HTMLParser

import choreographer


class SnapshotHTML(HTMLParser):
    """Remove executable/embedded content before handing markup to the renderer."""
    blocked = {"script", "iframe", "object", "embed", "base", "link", "meta"}
    void = {"area", "br", "col", "hr", "img", "input", "source", "wbr"}

    def __init__(self):
        super().__init__(convert_charrefs=False)
        self.parts = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.blocked:
            if tag in {"script", "iframe", "object"}:
                self.skip += 1
            return
        if self.skip:
            return
        clean = []
        for name, value in attrs:
            if name.startswith("on") or name in {"srcdoc", "action", "formaction", "srcset"}:
                continue
            if name in {"src", "href", "xlink:href"} and not (
                name == "src" and tag == "img" and (value or "").startswith("data:image/png;base64,")
            ):
                continue
            clean.append(name if value is None else f'{name}="{escape(value, quote=True)}"')
        self.parts.append("<" + tag + (" " + " ".join(clean) if clean else "") + ">")

    def handle_endtag(self, tag):
        if tag in {"script", "iframe", "object"}:
            self.skip = max(0, self.skip - 1)
        elif not self.skip and tag not in self.blocked and tag not in self.void:
            self.parts.append(f"</{tag}>")

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)

    def handle_entityref(self, name):
        if not self.skip:
            self.parts.append(f"&{name};")

    def handle_charref(self, name):
        if not self.skip:
            self.parts.append(f"&#{name};")


def snapshot_document(snapshot):
    markup = snapshot.get("markup", "")
    css = snapshot.get("css", "")
    if not isinstance(markup, str) or not isinstance(css, str) or len(markup) + len(css) > 25_000_000:
        raise ValueError("The dashboard snapshot is missing or too large.")
    width = max(720, min(2400, int(snapshot.get("width", 1280))))
    parser = SnapshotHTML()
    parser.feed(markup)
    # No scripts, external resources, source files or remote fonts are permitted.
    policy = "default-src 'none'; img-src data:; style-src 'unsafe-inline'; font-src data:; base-uri 'none'; form-action 'none'"
    css = css.replace("</", "<\\/")
    css += "html{background:#0b1420;}html:has(.theme-light){background:#fff;color-scheme:light;} .reader{break-inside:avoid;}"
    return (f'<!doctype html><html><head><meta charset="utf-8"><meta http-equiv="Content-Security-Policy" content="{policy}">'
            f'<title>ISBC dashboard</title><style>{css}</style></head><body>'
            + "".join(parser.parts) + "</body></html>"), width


async def _render_snapshot(document, width):
    # This isolated browser renders only the captured artifact, never the live app.
    async with choreographer.Browser() as browser:
        tab = browser.get_tab() or await browser.create_tab("about:blank")
        await tab.send_command("Page.enable")
        await tab.send_command("Emulation.setScriptExecutionDisabled", {"value": True})
        await tab.send_command("Network.enable")
        await tab.send_command("Network.setBlockedURLs", {"urls": ["http://*", "https://*", "file://*"]})
        await tab.send_command("Emulation.setDeviceMetricsOverride", {
            "width": width, "height": 1000, "deviceScaleFactor": 1, "mobile": False})
        await tab.send_command("Emulation.setEmulatedMedia", {"media": "screen"})
        tree = await tab.send_command("Page.getFrameTree")
        loaded = tab.subscribe_once("Page.loadEventFired")
        await tab.send_command("Page.setDocumentContent", {
            "frameId": tree["result"]["frameTree"]["frame"]["id"], "html": document})
        await asyncio.wait_for(loaded, timeout=30)
        # Match the original viewport width; page breaks preserve the dashboard layout.
        result = await tab.send_command("Page.printToPDF", {
            "printBackground": True, "paperWidth": width / 96,
            "paperHeight": width / 96 * 1.4142,
            "marginTop": .15, "marginBottom": .15, "marginLeft": 0, "marginRight": 0,
            "scale": 1, "displayHeaderFooter": False, "preferCSSPageSize": False})
        if "error" in result:
            raise RuntimeError(result["error"].get("message", "PDF rendering failed"))
        return base64.b64decode(result["result"]["data"])


def build_dashboard_pdf(snapshot):
    document, width = snapshot_document(snapshot)
    return asyncio.run(_render_snapshot(document, width))
