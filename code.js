(async () => {
  // ================== KONFIGURASI ==================
  const PENCARIAN = 'Obyek Berserial Nomor';
  const JENIS_OBYEK = 'Kendaraan Roda Empat';
  const NO_RANGKA  = 'MMBJNKB70ED045460';
  const NO_MESIN   = '4M40UAE8974';
  const TAHUN      = '2025'; // kamu bilang sudah keisi, jadi ini cuma fallback
  const MAX_WAIT_SELECT2 = 8000;  // 8 detik nunggu dropdown ke-2
  const MAX_WAIT_INPUT   = 15000; // 15 detik nunggu input muncul
  // =================================================

  const log = (...a) => console.log('[fidusia]', ...a);

  // helper tunggu kondisi
  function waitFor(checkFn, timeout = 5000, interval = 200) {
    return new Promise((resolve, reject) => {
      const start = Date.now();
      const timer = setInterval(() => {
        let ok = false;
        try { ok = checkFn(); } catch (e) {}
        if (ok) {
          clearInterval(timer);
          resolve(ok);
        } else if (Date.now() - start > timeout) {
          clearInterval(timer);
          reject(new Error('timeout'));
        }
      }, interval);
    });
  }

  // set select by visible text
  function setSelectByText(sel, text) {
    if (!sel) return false;
    const opt = [...sel.options].find(
      o => (o.text || '').trim().toLowerCase() === text.trim().toLowerCase()
    );
    if (!opt) return false;
    sel.value = opt.value;
    sel.dispatchEvent(new Event('input', { bubbles: true }));
    sel.dispatchEvent(new Event('change', { bubbles: true }));
    return true;
  }

  // setter “paksa” biar ke-detect framework
  function setRealValue(el, val) {
    const proto = Object.getPrototypeOf(el);
    const desc = Object.getOwnPropertyDescriptor(proto, 'value');
    if (desc && typeof desc.set === 'function') {
      desc.set.call(el, val);
    } else {
      el.value = val;
    }
  }

  // ================== STEP 1: pilih pencarian ==================
  const sel1 = document.querySelector('select');
  if (!sel1) {
    log('❗ select utama tidak ditemukan');
    return;
  }

  const ok1 = setSelectByText(sel1, PENCARIAN);
  if (!ok1) {
    log('❗ opsi "' + PENCARIAN + '" tidak ditemukan di select pertama');
    return;
  }
  log('✅ step 1 ok: pilih "' + PENCARIAN + '"');

  // ================== STEP 2: tunggu dan pilih dropdown ke-2 ==================
  let sel2;
  try {
    log('⏳ menunggu dropdown ke-2 terisi...');
    await waitFor(() => {
      const selects = [...document.querySelectorAll('select')].slice(1);
      sel2 = selects.find(s =>
        [...s.options].some(o => o.text.toLowerCase().includes('kendaraan'))
      );
      return !!sel2;
    }, MAX_WAIT_SELECT2, 300);
  } catch (e) {
    log('❗ dropdown ke-2 tidak muncul dalam ' + MAX_WAIT_SELECT2/1000 + ' detik');
    return;
  }

  const ok2 = setSelectByText(sel2, JENIS_OBYEK);
  if (!ok2) {
    log('❗ opsi "' + JENIS_OBYEK + '" tidak ditemukan di dropdown ke-2. Cek teks-nya persis.');
    return;
  }
  log('✅ step 2 ok: pilih "' + JENIS_OBYEK + '"');

  // ================== STEP 3: tunggu input muncul ==================
  // dari console kamu: id-nya cat_0_24 dan cat_0_25
  let inputRangka, inputMesin;
  try {
    log('⏳ menunggu input rangka & mesin muncul...');
    await waitFor(() => {
      inputRangka = document.getElementById('cat_0_24');
      inputMesin  = document.getElementById('cat_0_25');
      // pastikan visible
      const visR = inputRangka && inputRangka.offsetParent !== null;
      const visM = inputMesin && inputMesin.offsetParent !== null;
      return visR && visM;
    }, MAX_WAIT_INPUT, 300);
  } catch (e) {
    log('❗ input cat_0_24 / cat_0_25 tidak muncul dalam ' + MAX_WAIT_INPUT/1000 + ' detik');
    // bantu debug
    const allVisInputs = [...document.querySelectorAll('input')]
      .filter(i => i.offsetParent !== null)
      .map(i => ({
        id: i.id,
        name: i.name,
        type: i.type,
        placeholder: i.placeholder
      }));
    console.table(allVisInputs);
    return;
  }
  log('✅ step 3 ok: input sudah ada');

  // ================== STEP 4: isi pakai native setter + event lengkap ==================
  function fillAndTrigger(el, val) {
    if (!el) return;
    el.focus();
    setRealValue(el, val);
    el.dispatchEvent(new Event('input',  { bubbles: true }));
    el.dispatchEvent(new Event('change', { bubbles: true }));
    el.dispatchEvent(new Event('keyup',  { bubbles: true }));
    el.dispatchEvent(new Event('blur',   { bubbles: true }));
  }

  fillAndTrigger(inputRangka, NO_RANGKA);
  fillAndTrigger(inputMesin,  NO_MESIN);
  log('✅ step 4 ok: nomor rangka & mesin diisi');

  // ================== STEP 5: (opsional) pilih tahun 2025 kalau ada ==================
  const selTahun = [...document.querySelectorAll('select')].find(s =>
    [...s.options].some(o => o.text.trim() === TAHUN)
  );
  if (selTahun) {
    setSelectByText(selTahun, TAHUN);
    log('✅ tahun dipastikan ' + TAHUN);
  } else {
    log('ℹ️ select tahun tidak ditemukan / sudah terisi, lanjut...');
  }

  // ================== STEP 6: klik tombol cari ==================
  const btn = document.getElementById('submit-button')
    || [...document.querySelectorAll('button, input[type="submit"], input[type="button"]')]
       .find(b => (b.innerText || b.value || '').trim().toLowerCase() === 'cari');

  if (btn) {
    btn.click();
    log('✅ step 6 ok: tombol Cari diklik');
  } else {
    log('❗ tombol Cari tidak ketemu');
  }

  log('🎉 selesai, cek hasil di bawah form');
})();
