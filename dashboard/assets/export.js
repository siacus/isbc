/* Capture the current page, including live WebGL state, before PDF rendering. */
window.dash_clientside = Object.assign({}, window.dash_clientside, {
    dashboard: {
        capture: async function (clicks) {
            if (!clicks) return window.dash_clientside.no_update;
            try {
                const original = document.getElementById('app-shell');
                const graph = document.querySelector('#map .js-plotly-plot');
                if (!original) throw new Error('The dashboard has not loaded yet.');
                if (original.querySelector('[data-dash-is-loading="true"]')) {
                    throw new Error('Wait for the current selection to finish loading, then export again.');
                }
                const clone = original.cloneNode(true);
                // Browser properties such as checked/value/open are not all HTML attributes.
                const liveInputs = original.querySelectorAll('input, textarea, select, details');
                const copiedInputs = clone.querySelectorAll('input, textarea, select, details');
                liveInputs.forEach((source, i) => {
                    const target = copiedInputs[i];
                    if (source.tagName === 'DETAILS') {
                        target.toggleAttribute('open', source.open);
                    } else if (source.tagName === 'TEXTAREA') {
                        target.textContent = source.value;
                    } else if (source.tagName === 'INPUT') {
                        target.setAttribute('value', source.value);
                        target.toggleAttribute('checked', source.checked);
                    } else {
                        Array.from(target.options).forEach((option, j) => {
                            option.toggleAttribute('selected', source.options[j].selected);
                        });
                    }
                });
                // Flatten overflow only in the copy so the full current tables/text are exported.
                clone.querySelectorAll('.label-scroll, .table-scroll, .document-text').forEach(node => {
                    node.style.setProperty('max-height', 'none', 'important');
                    node.style.setProperty('height', 'auto', 'important');
                    node.style.setProperty('overflow', 'visible', 'important');
                    node.scrollTop = 0;
                });
                clone.querySelectorAll('script, iframe, object, embed, .modebar, .dash-spinner').forEach(n => n.remove());
                const status = clone.querySelector('#pdf-status');
                if (status) status.textContent = '';
                const button = clone.querySelector('#export-pdf');
                if (button) button.removeAttribute('disabled');
                if (graph && window.Plotly) {
                    const rect = graph.getBoundingClientRect();
                    const png = await window.Plotly.toImage(graph, {
                        format: 'png', width: Math.round(rect.width), height: Math.round(rect.height), scale: 2
                    });
                    const image = document.createElement('img');
                    image.src = png;
                    image.alt = 'Current binary-profile projection';
                    image.style.cssText = 'display:block;width:100%;height:auto;break-inside:avoid;';
                    const target = clone.querySelector('#map');
                    target.replaceChildren(image);
                    target.style.height = 'auto';
                    target.style.breakInside = 'avoid';
                }
                const sheets = [];
                for (const sheet of document.styleSheets) {
                    try {
                        for (const rule of sheet.cssRules) {
                            // The browser-print fallback hides controls; the snapshot must retain them.
                            if (rule.type === CSSRule.MEDIA_RULE && rule.conditionText === 'print') continue;
                            sheets.push(rule.cssText);
                        }
                    } catch (_) { /* no remote stylesheets are needed by this local app */ }
                }
                sheets.push('html,body{margin:0!important;} .app-shell{overflow:visible!important;} ' +
                    'thead{display:table-header-group;} tr,.label-card,.map{break-inside:avoid;} ' +
                    '.count-table thead th{position:static!important;} .reader{overflow:visible;} ' +
                    '.Select-menu-outer{display:none!important;}');
                return {markup: clone.outerHTML, css: sheets.join('\n'), width: document.documentElement.clientWidth,
                        capturedAt: new Date().toISOString()};
            } catch (error) {
                return {error: String(error.message || error), capturedAt: new Date().toISOString()};
            }
        }
    }
});
