const timer=document.getElementById("timer");

let t=0;
setInterval(()=>{
t++;
timer.innerText="Mission Time: "+t+" sec";
},1000);
