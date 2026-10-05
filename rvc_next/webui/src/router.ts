import { reactive } from 'vue';
import { createRouter, createWebHashHistory, type RouteRecordRaw } from 'vue-router';

declare module 'vue-router' {
  interface RouteMeta {
    /** The destination it belongs to, for the navigation and "started on another screen". */
    screen?: string;
  }
}

const SCREENS = ['convert', 'live', 'separate', 'train', 'models'];

/** The destination the app was last on (Preferences.lastScreen), opened when the address names none. */
function lastScreen(): string {
  try {
    const value = JSON.parse(localStorage.getItem('rvc-next:preferences') ?? '{}').lastScreen;
    return SCREENS.includes(value) ? value : 'convert';
  } catch {
    return 'convert';
  }
}

const routes: RouteRecordRaw[] = [
  { path: '/', redirect: () => `/${lastScreen()}` },
  { path: '/convert', name: 'convert', component: () => import('@/views/ConvertView.vue'), meta: { screen: 'convert' } },
  { path: '/live', name: 'live', component: () => import('@/views/LiveView.vue'), meta: { screen: 'live' } },
  { path: '/separate', name: 'separate', component: () => import('@/views/SeparateView.vue'), meta: { screen: 'separate' } },
  { path: '/train', name: 'train', component: () => import('@/views/TrainView.vue'), meta: { screen: 'train' } },
  { path: '/train/:name', name: 'experiment', component: () => import('@/views/ExperimentView.vue'), props: true, meta: { screen: 'train' } },
  { path: '/models', name: 'models', component: () => import('@/views/ModelsView.vue'), meta: { screen: 'models' } },
  { path: '/models/:id', name: 'model', component: () => import('@/views/ModelDetailView.vue'), props: true, meta: { screen: 'models' } },
  { path: '/jobs', name: 'jobs', component: () => import('@/views/JobsView.vue'), meta: { screen: 'jobs' } },
  { path: '/settings', name: 'settings', component: () => import('@/views/SettingsView.vue'), meta: { screen: 'settings' } },
  { path: '/dev', name: 'dev', component: () => import('@/views/DevComponentsView.vue'), meta: { screen: 'dev' } },
  { path: '/:pathMatch(.*)*', redirect: '/convert' },
];

// Hash history: the UI works at any deployment sub-path with no server-side rewrites.
export const router = createRouter({ history: createWebHashHistory(), routes });

// The address each destination was last at, so the navigation goes back to the experiment or the
// voice it left rather than to its list. Memory only.
const lastAddress = reactive<Record<string, string>>({});
router.afterEach((to) => {
  if (to.meta.screen) lastAddress[to.meta.screen] = to.fullPath;
});

export const addressOf = (screen: string): string => lastAddress[screen] ?? `/${screen}`;
