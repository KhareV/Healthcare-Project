<script lang="ts">
	import { Canvas, T } from '@threlte/core';
	import { GLTF } from '@threlte/extras';
	import { onMount } from 'svelte';
	import { Box3, Vector3 } from 'three';

	// The heart model's native units are arbitrary (~100 across), so measure
	// the loaded scene and normalise it to a fixed on-screen size, centred.
	const TARGET_SIZE = 3.6;
	let fitScale = 0.04;
	let fitOffset: [number, number, number] = [0, 0, 0];
	function fitModel(gltf: { scene: import('three').Object3D }) {
		const box = new Box3().setFromObject(gltf.scene);
		const size = box.getSize(new Vector3());
		const center = box.getCenter(new Vector3());
		const longest = Math.max(size.x, size.y, size.z) || 1;
		fitScale = TARGET_SIZE / longest;
		fitOffset = [-center.x * fitScale, -center.y * fitScale, -center.z * fitScale];
	}

	// Gentle, continuous rotation for an ultra-premium feel
	let rotY = 0;
	let rotX = 0.15;
	
	onMount(() => {
		let animationId: number;
		const animate = () => {
			rotY -= 0.002; // Smooth slow rotation
			rotX = 0.15 + Math.sin(Date.now() * 0.001) * 0.05; // Gentle float
			animationId = requestAnimationFrame(animate);
		};
		animate();
		return () => cancelAnimationFrame(animationId);
	});
</script>

<div class="heart-scene">
	<Canvas dpr={[1, 2]}>
		<!-- Camera positioned so the full heart fits gracefully -->
		<T.PerspectiveCamera makeDefault position={[0, 0.2, 9.8]} fov={32} />

		<!-- Premium Studio Lighting Setup -->
		<T.AmbientLight intensity={1.4} color="#ffffff" />
		<T.DirectionalLight position={[5, 6, 5]} intensity={4.5} color="#14b8a6" />
		<T.DirectionalLight position={[-5, 4, -2]} intensity={3} color="#38bdf8" />
		<T.SpotLight position={[0, 8, 2]} intensity={7} color="#ffffff" angle={0.6} penumbra={1} />
		<T.PointLight position={[0, -3, 3]} intensity={2.5} color="#2dd4bf" />

		<!-- Centered and scaled model -->
		<T.Group rotation={[rotX, rotY, -0.05]}>
			<T.Group scale={fitScale} position={fitOffset}>
				<GLTF url="/models/human-heart.glb" onload={fitModel} />
			</T.Group>
		</T.Group>
	</Canvas>
</div>

<style>
	.heart-scene {
		position: absolute;
		/* Let the heart breathe beyond the device column. The canvas is
		   intentionally oversized so it can cross the surrounding grid
		   and labels like a physical object in front of the page. */
		top: -5%;
		left: -20%;
		width: 160%;
		height: 110%;
		z-index: 30;
		transform: translateY(-4%);
		pointer-events: none;
		overflow: visible;
	}

	.heart-scene :global(canvas) {
		display: block;
		width: 100% !important;
		height: 100% !important;
		outline: none;
		background: transparent !important;
	}

	@media (max-width: 1050px) {
		.heart-scene { left: -5%; width: 110%; height: 104%; }
	}
</style>
