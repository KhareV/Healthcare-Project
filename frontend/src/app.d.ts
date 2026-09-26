// See https://svelte.dev/docs/kit/types#app.d.ts
// for information about these interfaces
import 'svelte-clerk/env';

declare global {
	namespace App {
		// interface Error {}
		interface Locals {
			authConfigured: boolean;
		}
		// interface PageData {}
		// interface PageState {}
		// interface Platform {}
	}
}

export {};
