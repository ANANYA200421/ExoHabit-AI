const c=document.createElement("canvas");
document.body.appendChild(c);
c.style.position="fixed";c.style.zIndex=-1;
const x=c.getContext("2d");

function r(){c.width=innerWidth;c.height=innerHeight}
r();onresize=r;

let s=[];
for(let i=0;i<200;i++)
s.push({x:Math.random()*c.width,y:Math.random()*c.height,r:Math.random()*2});

(function draw(){
x.clearRect(0,0,c.width,c.height);
x.fillStyle="white";
s.forEach(a=>{x.beginPath();x.arc(a.x,a.y,a.r,0,6.28);x.fill();});
requestAnimationFrame(draw);
})();
