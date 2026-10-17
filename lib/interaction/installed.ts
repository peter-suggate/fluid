import { createInteractionCatalog } from './catalog';
import { character, characterMotor, characterRuntime } from './features/character/definition';
import { rigidProp, rigidBody, rigidRuntime } from './features/rigid-prop/definition';
import { sceneSwitch, switchToggle, switchRuntime, staticBody, staticBodyRuntime } from './features/switch/definition';
import { movingPlatform, platformMotion, platformRuntime } from './features/moving-platform/definition';
import { triggerVolume, triggerSensor, triggerRuntime } from './features/trigger-volume/definition';
export const interactionCatalog = createInteractionCatalog({ entities: [character, rigidProp, sceneSwitch, movingPlatform, triggerVolume], behaviors: [characterMotor, rigidBody, staticBody, switchToggle, platformMotion, triggerSensor] });
export const interactionRuntimes = [characterRuntime, rigidRuntime, staticBodyRuntime, switchRuntime, platformRuntime, triggerRuntime];
