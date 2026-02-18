const form = document.getElementById("predictForm");

if (form) {
    form.addEventListener("submit", async (e) => {
        e.preventDefault();

        const data = {
            pl_rade: form.pl_rade.value,
            pl_eqt: form.pl_eqt.value,
            pl_orbper: form.pl_orbper.value,
            st_teff: form.st_teff.value,
            stellar_compatibility: form.stellar_compatibility.value
        };

        const response = await fetch("http://127.0.0.1:5000/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(data)
        });

        const result = await response.json();

        document.getElementById("result").innerHTML = `
            <div class="card p-3">
                <h3>${result.habitability_class}</h3>
                <p>Probability: ${result.probability}</p>
                <p>Confidence: ${result.confidence_percent}%</p>
            </div>
        `;
    });
}
