async function predictHabitability() {

    const radius = document.getElementById("radius").value;
    const mass = document.getElementById("mass").value;
    const period = document.getElementById("period").value;
    const temp = document.getElementById("temp").value;
    const star = document.getElementById("starType").value;

    if (!radius || !mass || !period || !temp) {
        alert("Fill all fields");
        return;
    }

    const response = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            planet_radius: radius,
            planet_mass: mass,
            orbital_period: period,
            equilibrium_temp: temp,
            star_type: star
        })
    });

    const data = await response.json();

    const out = document.getElementById("result");

    if (data.status === "success") {
        out.innerHTML =
            `<h2>${data.habitable}</h2>
             <p>Score: ${data.score.toFixed(3)}</p>`;
    } else {
        out.innerText = data.message;
    }
}
