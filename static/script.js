// static/script.js
document.getElementById('run').addEventListener('click', async () => {
    const fileEl = document.getElementById('file');
    const model = document.getElementById('model').value;
    const rebal = document.getElementById('rebal').value;
    const lookback = document.getElementById('lookback').value;
  
    const form = new FormData();
    form.append('model', model);
    form.append('rebal', rebal);
    form.append('lookback', lookback);
  
    if (fileEl.files.length > 0) {
      form.append('file', fileEl.files[0]);
    } else {
      alert('CSV 파일을 업로드하거나 샘플 데이터를 로드해 주세요.');
      return;
    }
  
    document.getElementById('output').textContent = '계산 중...';
  
    try {
      const res = await fetch('/api/calculate', { method: 'POST', body: form });
      const data = await res.json();
      if (!res.ok) {
        document.getElementById('output').textContent = '에러: ' + JSON.stringify(data);
        return;
      }
      document.getElementById('output').textContent = JSON.stringify(data, null, 2);
    } catch (e) {
      document.getElementById('output').textContent = '통신 오류: ' + e.toString();
    }
  });
  
  // 샘플 데이터 버튼: 간단한 랜덤 returns 만들기 (CSV 대신 POST JSON 사용)
  document.getElementById('use-sample').addEventListener('click', async () => {
    const assets = ['A', 'B', 'C', 'D'];
    const dates = [];
    const now = new Date();
    for (let i = 0; i < 252; i++) {
      const d = new Date(now);
      d.setDate(now.getDate() - (252 - i));
      dates.push(d.toISOString().slice(0,10));
    }
    const returns = {};
    assets.forEach(a => {
      returns[a] = [];
    });
    for (let i = 0; i < 252; i++) {
      assets.forEach(a => {
        // random daily return ~ N(0.0005, 0.01)
        returns[a].push((Math.random()-0.5)*0.02);
      });
    }
    // build CSV text
    let csv = 'date,' + assets.join(',') + '\n';
    for (let i=0;i<dates.length;i++){
      const row = [dates[i]];
      assets.forEach(a => row.push(returns[a][i].toFixed(6)));
      csv += row.join(',') + '\n';
    }
  
    // create a fake file and set to file input using DataTransfer
    const blob = new Blob([csv], { type: 'text/csv' });
    const file = new File([blob], 'sample_returns.csv', { type: 'text/csv' });
    const dt = new DataTransfer();
    dt.items.add(file);
    document.getElementById('file').files = dt.files;
    alert('샘플 데이터가 업로드 되었습니다. 이제 "계산 실행"을 눌러주세요.');
  });
  