for(let i=0;i<200;i++){
let s=document.createElement("div");
s.style.position="fixed";
s.style.width="2px";
s.style.height="2px";
s.style.background="cyan";
s.style.left=Math.random()*100+"%";
s.style.top=Math.random()*100+"%";
s.style.opacity=Math.random();
document.body.appendChild(s);
}
