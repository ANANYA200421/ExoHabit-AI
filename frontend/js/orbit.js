const sat=document.createElement("div");
sat.style.width="8px";
sat.style.height="8px";
sat.style.background="white";
sat.style.position="fixed";
sat.style.borderRadius="50%";
document.body.appendChild(sat);

let a=0;
function orbit(){
a+=0.01;
sat.style.left=300+Math.cos(a)*120+"px";
sat.style.top=250+Math.sin(a)*120+"px";
requestAnimationFrame(orbit);
}
orbit();
