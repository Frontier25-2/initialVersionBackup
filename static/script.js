// static/script.js
document.getElementById("run").addEventListener("click", async () => {
  const fileInput = document.getElementById("file");
  const model = document.getElementById("model").value;
  const rebal = document.getElementById("rebal").value;
  const lookback = document.getElementById("lookback").value;

  const formData = new FormData();
  formData.append("model", model);
  formData.append("rebal", rebal);
  formData.append("lookback", lookback);

  if (fileInput.files.length > 0) {
    formData.append("file", fileInput.files[0]);
  }

  document.getElementById("output").textContent = "계산 중...";

  const res = await fetch("/api/calculate", {
    method: "POST",
    body: formData
  });

  const data = await res.json();

  if (data.error) {
    document.getElementById("output").textContent = `에러: ${data.error}`;
    return;
  }

  // 결과 텍스트 출력
  let text = "📊 모델 결과\n\n";
  text += "가중치:\n" + JSON.stringify(data.weights, null, 2) + "\n\n";
  text += "메트릭:\n" + JSON.stringify(data.metrics, null, 2);
  document.getElementById("output").textContent = text;

  // 그래프 추가 표시
  if (data.plot_img) {
    const img = document.createElement("img");
    img.src = "data:image/png;base64," + data.plot_img;
    img.alt = "누적수익률 그래프";
    img.style.maxWidth = "100%";
    img.style.display = "block";
    img.style.marginTop = "1em";
    document.getElementById("output").after(img);
  }
});
