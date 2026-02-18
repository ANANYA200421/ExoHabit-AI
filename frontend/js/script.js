async function predict(){

const data={
pl_rade:parseFloat(document.getElementById("pl_rade").value),
pl_eqt:parseFloat(document.getElementById("pl_eqt").value),
pl_orbper:parseFloat(document.getElementById("pl_orbper").value),
st_teff:parseFloat(document.getElementById("st_teff").value)
};

const res=await fetch("/predict",{
method:"POST",
headers:{"Content-Type":"application/json"},
body:JSON.stringify(data)
});

const result=await res.json();

document.getElementById("result").innerText=result.classification;
document.getElementById("score").innerText="Score: "+result.habitability_score;
}
