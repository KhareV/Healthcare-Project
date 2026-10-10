<script lang="ts">
	// Sortable evidence table. Cells are pre-formatted by the caller (fmt) so units stay explicit.
	type Col = { key: string; label: string; fmt?: (v: any, row: Record<string, any>) => string; align?: 'left' | 'right'; sortable?: boolean };
	let {
		columns,
		rows,
		caption = '',
		defaultSort = '',
		rowClass
	}: { columns: Col[]; rows: Record<string, any>[]; caption?: string; defaultSort?: string; rowClass?: (row: Record<string, any>) => string } = $props();

	let sortKey = $state('');
	let sortDir = $state<1 | -1>(1);
	$effect(() => {
		if (!sortKey && defaultSort) sortKey = defaultSort;
	});
	const sorted = $derived.by(() => {
		if (!sortKey) return rows;
		return [...rows].sort((a, b) => {
			const x = a[sortKey], y = b[sortKey];
			if (typeof x === 'number' && typeof y === 'number') return (x - y) * sortDir;
			return String(x ?? '').localeCompare(String(y ?? '')) * sortDir;
		});
	});
	function toggle(key: string) {
		if (sortKey === key) sortDir = (sortDir * -1) as 1 | -1;
		else {
			sortKey = key;
			sortDir = 1;
		}
	}
</script>

<div class="table-wrap">
	<table>
		{#if caption}<caption>{caption}</caption>{/if}
		<thead>
			<tr>
				{#each columns as col}
					<th class:right={col.align === 'right'} aria-sort={sortKey === col.key ? (sortDir === 1 ? 'ascending' : 'descending') : 'none'}>
						{#if col.sortable === false}{col.label}{:else}<button type="button" onclick={() => toggle(col.key)}>{col.label}{#if sortKey === col.key}<i>{sortDir === 1 ? '▲' : '▼'}</i>{/if}</button>{/if}
					</th>
				{/each}
			</tr>
		</thead>
		<tbody>
			{#each sorted as row}
				<tr class={rowClass ? rowClass(row) : ''}>
					{#each columns as col}
						<td class:right={col.align === 'right'}>{col.fmt ? col.fmt(row[col.key], row) : (row[col.key] ?? '—')}</td>
					{/each}
				</tr>
			{/each}
		</tbody>
	</table>
</div>

<style>
	.table-wrap{max-width:100%;overflow-x:auto;border:1px solid rgba(148,163,184,.12)}
	table{width:100%;border-collapse:collapse;font:10px 'JetBrains Mono',monospace;color:#b7c6d4}
	caption{padding:9px 12px;text-align:left;color:#53647b;font-size:8px;letter-spacing:.1em}
	th,td{padding:8px 11px;border-bottom:1px solid rgba(148,163,184,.08);text-align:left;white-space:nowrap}
	th{background:rgba(10,18,34,.9);color:#71829a;font-weight:500;font-size:8px;letter-spacing:.09em;text-transform:uppercase}
	th button{all:unset;cursor:pointer;display:inline-flex;gap:5px;align-items:center}
	th button:focus-visible{outline:2px solid #38bdf8}
	th i{font-style:normal;color:#2bb8b0;font-size:7px}
	.right{text-align:right}
	tr.flag td{color:#fda4af}
	tr.best td{color:#eef7f6}
	tr.privileged td{color:#c4b5fd}
	tr:hover td{background:rgba(43,184,176,.04)}
</style>
