fetch("/dataset").then(r=>r.json()).then(d=>{
new Chart(chart,{
type:"doughnut",
data:{
labels:["Habitable","Non"],
datasets:[{data:[d.habitable,d.total-d.habitable]}]
}
});
});
