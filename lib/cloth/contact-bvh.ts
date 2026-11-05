/** Swept AABB broadphase. Bounds contain both endpoints plus physical radius;
 * narrowphase still uses conservative continuous vertex/face and edge CCD. */
export class ClothContactBVH {
    private readonly order: number[];
    private readonly nodes: {
        min: number[];
        max: number[];
        left: number;
        right: number;
        start: number;
        end: number;
    }[] = [];
    constructor(readonly bounds: Float64Array) {
        if (bounds.length % 6)
            throw new Error('Invalid cloth contact bounds');
        this.order = Array.from({ length: bounds.length / 6 }, (_, i) => i);
        if (this.order.length)
            this.build(0, this.order.length);
    }
    /** Refit preserves exact broadphase bounds through arbitrary deformation;
     * topology remains a search partition, never an approximation of contact. */
    refit(): void {
        for (let i = this.nodes.length - 1; i >= 0; i--) {
            const node = this.nodes[i]!;
            node.min.fill(Infinity); node.max.fill(-Infinity);
            if (node.left >= 0) {
                const left = this.nodes[node.left]!, right = this.nodes[node.right]!;
                for (let a = 0; a < 3; a++) { node.min[a] = Math.min(left.min[a]!, right.min[a]!); node.max[a] = Math.max(left.max[a]!, right.max[a]!); }
            } else for (let j = node.start; j < node.end; j++) {
                const at = 6 * this.order[j]!;
                for (let a = 0; a < 3; a++) { node.min[a] = Math.min(node.min[a]!, this.bounds[at + a]!); node.max[a] = Math.max(node.max[a]!, this.bounds[at + a + 3]!); }
            }
        }
    }
    private build(start: number, end: number): number {
        const min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
        for (let i = start; i < end; i++) {
            const at = 6 * this.order[i]!;
            for (let a = 0; a < 3; a++) {
                min[a] = Math.min(min[a]!, this.bounds[at + a]!);
                max[a] = Math.max(max[a]!, this.bounds[at + a + 3]!);
            }
        }
        const index = this.nodes.length, node = { min, max, left: -1, right: -1, start, end };
        this.nodes.push(node);
        if (end - start > 8) {
            let axis = 0;
            for (let a = 1; a < 3; a++)
                if (max[a]! - min[a]! > max[axis]! - min[axis]!)
                    axis = a;
            // Median split keeps pathological coplanar cloth balanced.
            const sorted = this.order.slice(start, end).sort((a, b) => (this.bounds[6 * a + axis]! + this.bounds[6 * a + axis + 3]!) - (this.bounds[6 * b + axis]! + this.bounds[6 * b + axis + 3]!) || a - b);
            for (let i = 0; i < sorted.length; i++)
                this.order[start + i] = sorted[i]!;
            const mid = (start + end) >>> 1;
            node.left = this.build(start, mid);
            node.right = this.build(mid, end);
        }
        return index;
    }
    query(min: readonly number[], max: readonly number[]): number[] {
        if (!this.nodes.length)
            return [];
        const result: number[] = [], stack = [0];
        while (stack.length) {
            const node = this.nodes[stack.pop()!]!;
            if (node.min.some((v, a) => v > max[a]! || node.max[a]! < min[a]!))
                continue;
            if (node.left >= 0) {
                stack.push(node.right, node.left);
                continue;
            }
            for (let j = node.start; j < node.end; j++) {
                const id = this.order[j]!, at = 6 * id;
                let overlap = true;
                for (let a = 0; a < 3; a++)
                    if (this.bounds[at + a]! > max[a]! || this.bounds[at + a + 3]! < min[a]!) {
                        overlap = false;
                        break;
                    }
                if (overlap)
                    result.push(id);
            }
        }
        return result.sort((a, b) => a - b);
    }
}
