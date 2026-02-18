const tele=document.getElementById("telemetry");

setInterval(()=>{
tele.innerText=
"Signal Strength : "+Math.floor(Math.random()*100)+" %\n"+
"Velocity        : "+(Math.random()*20).toFixed(2)+" km/s\n"+
"Radiation Level : "+Math.floor(Math.random()*5)+"\n"+
"Power Output    : "+Math.floor(Math.random()*100)+" %";
},1000);
