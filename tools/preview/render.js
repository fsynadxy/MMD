// Usage: node render.js <http base> <model url path> <three url path> <out dir> <views.json>
const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');
(async () => {
  const [base, model, three, outDir, viewsFile] = process.argv.slice(2);
  const views = JSON.parse(fs.readFileSync(viewsFile, 'utf8'));
  fs.mkdirSync(outDir, { recursive: true });
  const browser = await chromium.launch({ args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'] });
  const W = views.width || 900, H = views.height || 1200;
  const page = await browser.newPage({ viewport: { width: W, height: H } });
  page.on('console', m => { if (m.type() === 'error' || m.type() === 'warning') console.log('[page]', m.text()); });
  page.on('pageerror', e => console.log('[pageerror]', e.message));
  const url = `${base}/tools/preview/view.html?model=${encodeURIComponent(model)}&three=${encodeURIComponent(three)}&w=${W}&h=${H}&amb=${views.amb||2.0}&dir=${views.dir||3.0}`;
  await page.goto(url);
  await page.waitForFunction(() => window.state && (window.state.ready || window.state.error), null, { timeout: 120000 });
  const st = await page.evaluate(() => window.state);
  if (st.error) { console.log('ERROR', st.error); process.exit(1); }
  console.log('loaded', JSON.stringify(st.info));
  // wait until every texture (map / sphere / toon) has an image
  await page.waitForFunction(() => window.mesh.material.every(m => ['map', 'matcap', 'gradientMap'].every(k => !m[k] || (m[k].image && (m[k].image.width > 0 || m[k].image.naturalWidth > 0)))), null, { timeout: 60000 });
  await page.waitForTimeout(300);
  for (const v of views.views) {
    if (v.physics !== undefined) {
      if (v.physics > 0) {
        const r = await page.evaluate(([s, rot]) => window.runPhysics(s, rot), [v.physics, v.physicsRot || null]);
        console.log('physics', JSON.stringify(r));
      }
      await page.evaluate(v2 => window.renderCurrent(v2), v);
    } else {
      await page.evaluate(v2 => window.renderView(v2), v);
    }
    const file = path.join(outDir, v.name + '.png');
    await page.locator('canvas').screenshot({ path: file });
    console.log('wrote', file);
  }
  await browser.close();
})().catch(e => { console.error(e); process.exit(1); });
