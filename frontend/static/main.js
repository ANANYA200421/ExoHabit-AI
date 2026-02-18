document.getElementById("predictForm").addEventListener("submit", async function(e) {
    e.preventDefault();

    const data = {
        pl_rade: document.getElementById("pl_rade").value,
        pl_eqt: document.getElementById("pl_eqt").value,
        pl_orbper: document.getElementById("pl_orbper").value,
        st_teff: document.getElementById("st_teff").value,
        stellar_compatibility: document.getElementById("stellar_compatibility").value
    };

    const response = await fetch("http://127.0.0.1:5000/api/predict", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify(data)
    });

    const result = await response.json();

    document.getElementById("result").innerHTML =
        "Prediction: " + result.habitability_class +
        "<br>Probability: " + result.probability.toFixed(2);
});
