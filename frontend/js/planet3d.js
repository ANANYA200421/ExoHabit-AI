const container=document.getElementById("planet");

const scene=new THREE.Scene();
const camera=new THREE.PerspectiveCamera(75,container.clientWidth/container.clientHeight,0.1,1000);

const renderer=new THREE.WebGLRenderer({alpha:true});
renderer.setSize(container.clientWidth,container.clientHeight);
container.appendChild(renderer.domElement);

const geometry=new THREE.SphereGeometry(2,64,64);

const texture=new THREE.TextureLoader().load(
"https://threejs.org/examples/textures/planets/earth_atmos_2048.jpg"
);

const material=new THREE.MeshStandardMaterial({
map:texture,
emissive:0x00ffff,
emissiveIntensity:0.2
});

const planet=new THREE.Mesh(geometry,material);
scene.add(planet);

const light=new THREE.PointLight(0xffffff,1);
light.position.set(5,3,5);
scene.add(light);

camera.position.z=5;

function animate(){
requestAnimationFrame(animate);
planet.rotation.y+=0.002;
renderer.render(scene,camera);
}
animate();
