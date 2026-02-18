const log=document.getElementById("log");

setInterval(()=>{
const msg=
new Date().toLocaleTimeString()+
"  telemetry packet received\n";

log.textContent+=msg;
log.scrollTop=log.scrollHeight;

},1200);
