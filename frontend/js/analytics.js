async function loadAnalytics(){

const res = await fetch("/analytics");
const data = await res.json();

Chart.defaults.color = "#cfe9ff";

/* =====================================================
   1️⃣ FEATURE IMPORTANCE — HORIZONTAL BAR (ML STYLE)
===================================================== */

new Chart(document.getElementById("importanceChart"),{
type:"bar",
data:{
labels:data.feature_names,
datasets:[{
data:data.feature_importance,
backgroundColor:"#3fa7ff"
}]
},
options:{
indexAxis:"y",
plugins:{
legend:{display:false}
},
scales:{
x:{title:{display:true,text:"Importance Weight"}}
}
}
});


/* =====================================================
   2️⃣ HABITABILITY DISTRIBUTION — DOUGHNUT + %
===================================================== */

const total = data.habitable + data.non_habitable;

new Chart(document.getElementById("habitChart"),{
type:"doughnut",
data:{
labels:[
"Habitable ("+Math.round(data.habitable/total*100)+"%)",
"Non-Habitable ("+Math.round(data.non_habitable/total*100)+"%)"
],
datasets:[{
data:[data.habitable,data.non_habitable],
backgroundColor:["#00ffa6","#ff4d6d"]
}]
},
options:{
cutout:"65%"
}
});


/* =====================================================
   3️⃣ STAR–PLANET RELATIONSHIP — COLOR SCATTER
===================================================== */

const habitablePoints = [];
const nonHabitablePoints = [];

for(let i=0;i<data.star_temp.length;i++){

const point = {
x:data.star_temp[i],
y:data.planet_temp[i]
};

if(data.labels[i] === 1)
habitablePoints.push(point);
else
nonHabitablePoints.push(point);
}

new Chart(document.getElementById("scatterChart"),{
type:"scatter",
data:{
datasets:[
{
label:"Habitable",
data:habitablePoints,
backgroundColor:"#00ffa6"
},
{
label:"Non-Habitable",
data:nonHabitablePoints,
backgroundColor:"#ff4d6d"
}
]
},
options:{
plugins:{
legend:{position:"top"}
},
scales:{
x:{
title:{display:true,text:"Star Temperature (K)"}
},
y:{
title:{display:true,text:"Planet Temperature (K)"}
}
}
}
});

}

loadAnalytics();
