// Knowledge Graph Visualization with D3.js

class KnowledgeGraphVisualization {
    constructor() {
        this.width = window.innerWidth; // Full screen width
        this.height = window.innerHeight;
        this.nodes = [];
        this.links = [];
        this.simulation = null;
        this.svg = null;
        this.g = null;
        this.showLabels = true;
        this.showPassages = true;
        this.selectedNode = null;
        this.searchCache = {};
        
        this.nodeColors = {
            'Concept': '#1f77b4',
            'Dataset': '#ff7f0e',
            'Passage': '#2ca02c',
            'Document': '#d62728',
            'Unknown': '#7f7f7f'
        };
        
        this.edgeColors = {
            'RELATED_TO': '#666',
            'SUBCONCEPT_OF': '#1f77b4',
            'HAS_SUBCONCEPT': '#1f77b4',
            'MENTIONS': '#2ca02c',
            'HAS_DATASET_ABOUT': '#ff7f0e',
            'DATASET_ON_TOPIC': '#ff7f0e'
        };
        
        this.init();
    }
    
    init() {
        // Initialize SVG
        this.svg = d3.select('#graph')
            .attr('width', this.width)
            .attr('height', this.height);
        
        // Add zoom behavior
        const zoom = d3.zoom()
            .scaleExtent([0.1, 10])
            .on('zoom', (event) => {
                this.g.attr('transform', event.transform);
            });
        
        this.svg.call(zoom);
        
        // Create main group
        this.g = this.svg.append('g');
        
        // Create groups for links and nodes
        this.linkGroup = this.g.append('g').attr('class', 'links');
        this.nodeGroup = this.g.append('g').attr('class', 'nodes');
        
        // Initialize force simulation
        this.simulation = d3.forceSimulation()
            .force('link', d3.forceLink().id(d => d.id).distance(100))
            .force('charge', d3.forceManyBody().strength(-300))
            .force('center', d3.forceCenter(this.width / 2, this.height / 2))
            .force('collision', d3.forceCollide().radius(30));
        
        // Load initial data
        this.loadTopConcepts();
        
        // Set up event listeners
        this.setupEventListeners();
        
        // Handle window resize
        window.addEventListener('resize', () => this.handleResize());
    }
    
    setupEventListeners() {
        // Minimal event listeners for graph interaction
        // Most interaction is handled in the graph itself
    }
    
    async loadTopConcepts() {
        try {
            const response = await fetch('/api/kg/top_concepts?limit=10');
            const data = await response.json();
            
            // Create initial nodes from top concepts
            const initialNodes = data.concepts.map(concept => ({
                id: concept.id,
                label: concept.label,
                kind: 'Concept',
                centrality: concept.centrality,
                degree: concept.degree
            }));
            
            // Load subgraph for the most central concept
            if (initialNodes.length > 0) {
                await this.loadSubgraph(initialNodes[0].id, 2);
            }
        } catch (error) {
            console.error('Error loading top concepts:', error);
        }
    }
    
    async loadStats() {
        try {
            const response = await fetch('/api/kg/stats');
            const stats = await response.json();
            
            const statsHtml = `
                <div class="stat-item">Total Nodes: ${stats.total_nodes}</div>
                <div class="stat-item">Total Edges: ${stats.total_edges}</div>
                ${Object.entries(stats.node_counts).map(([kind, count]) => 
                    `<div class="stat-item">${kind}: ${count}</div>`
                ).join('')}
            `;
            
            document.getElementById('stats').innerHTML = statsHtml;
            
            // Update edge legend
            const edgeLegendHtml = stats.edge_types.map(type => `
                <div class="legend-item">
                    <svg width="40" height="2">
                        <line x1="0" y1="1" x2="40" y2="1" stroke="${this.edgeColors[type] || '#999'}" stroke-width="2"></line>
                    </svg>
                    <span>${type.replace(/_/g, ' ')}</span>
                </div>
            `).join('');
            
            document.getElementById('edgeLegend').innerHTML = edgeLegendHtml;
        } catch (error) {
            console.error('Error loading stats:', error);
        }
    }
    
    async loadSubgraph(nodeId, depth = 1) {
        try {
            const response = await fetch('/api/kg/subgraph', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    node_id: nodeId, 
                    depth: depth,
                    max_nodes: 50 
                })
            });
            
            const data = await response.json();
            this.updateData(data.nodes, data.edges);
            
            // Select the center node
            this.selectNode(nodeId);
        } catch (error) {
            console.error('Error loading subgraph:', error);
        }
    }
    
    // Pure graph visualization - no UI interactions needed
    // All concepts and relationships data is returned via API only
    // No query relevance highlighting - all returned concepts are relevant
    
    updateData(newNodes, newLinks) {
        // Filter nodes and links based on settings
        if (!this.showPassages) {
            newNodes = newNodes.filter(n => n.kind !== 'Passage');
            newLinks = newLinks.filter(l => {
                const sourceNode = newNodes.find(n => n.id === l.source);
                const targetNode = newNodes.find(n => n.id === l.target);
                return sourceNode && targetNode;
            });
        }
        
        // Merge with existing data
        const nodeMap = new Map(this.nodes.map(n => [n.id, n]));
        
        newNodes.forEach(node => {
            if (!nodeMap.has(node.id)) {
                nodeMap.set(node.id, node);
            }
        });
        
        this.nodes = Array.from(nodeMap.values());
        
        // Update links
        const linkSet = new Set(this.links.map(l => `${l.source.id || l.source}-${l.target.id || l.target}`));
        
        newLinks.forEach(link => {
            const linkId = `${link.source}-${link.target}`;
            if (!linkSet.has(linkId)) {
                this.links.push(link);
            }
        });
        
        this.updateGraph();
    }
    
    updateGraph() {
        // Update links
        const link = this.linkGroup.selectAll('.link')
            .data(this.links, d => `${d.source.id || d.source}-${d.target.id || d.target}`);
        
        link.exit().remove();
        
        const linkEnter = link.enter()
            .append('line')
            .attr('class', 'link')
            .style('stroke', d => this.edgeColors[d.type] || '#999')
            .on('mouseover', (event, d) => this.showTooltip(event, `${d.type.replace(/_/g, ' ')}`))
            .on('mouseout', () => this.hideTooltip());
        
        // Update nodes
        const node = this.nodeGroup.selectAll('.node-group')
            .data(this.nodes, d => d.id);
        
        node.exit().remove();
        
        const nodeEnter = node.enter()
            .append('g')
            .attr('class', 'node-group')
            .call(d3.drag()
                .on('start', (event, d) => this.dragStarted(event, d))
                .on('drag', (event, d) => this.dragged(event, d))
                .on('end', (event, d) => this.dragEnded(event, d)));
        
        // Add shapes based on node kind
        nodeEnter.each((d, i, nodes) => {
            const g = d3.select(nodes[i]);
            
            if (d.kind === 'Concept') {
                g.append('circle')
                    .attr('class', 'node')
                    .attr('r', 15)
                    .style('fill', this.nodeColors[d.kind]);
            } else if (d.kind === 'Dataset') {
                g.append('rect')
                    .attr('class', 'node')
                    .attr('x', -15)
                    .attr('y', -15)
                    .attr('width', 30)
                    .attr('height', 30)
                    .style('fill', this.nodeColors[d.kind]);
            } else if (d.kind === 'Passage') {
                g.append('polygon')
                    .attr('class', 'node')
                    .attr('points', '0,-15 13,7.5 -13,7.5')
                    .style('fill', this.nodeColors[d.kind]);
            } else {
                g.append('circle')
                    .attr('class', 'node')
                    .attr('r', 10)
                    .style('fill', this.nodeColors.Unknown);
            }
        });
        
        // Add labels
        nodeEnter.append('text')
            .attr('class', 'node-label')
            .attr('dy', 25)
            .attr('text-anchor', 'middle')
            .style('display', this.showLabels ? 'block' : 'none')
            .text(d => d.label.length > 20 ? d.label.substring(0, 20) + '...' : d.label);
        
        // Add click handlers
        nodeEnter.on('click', (event, d) => this.handleNodeClick(event, d))
            .on('mouseover', (event, d) => this.showNodeTooltip(event, d))
            .on('mouseout', () => this.hideTooltip());
        
        // Update simulation
        this.simulation.nodes(this.nodes);
        this.simulation.force('link').links(this.links);
        this.simulation.alpha(0.3).restart();
        
        this.simulation.on('tick', () => {
            this.linkGroup.selectAll('.link')
                .attr('x1', d => d.source.x)
                .attr('y1', d => d.source.y)
                .attr('x2', d => d.target.x)
                .attr('y2', d => d.target.y);
            
            this.nodeGroup.selectAll('.node-group')
                .attr('transform', d => `translate(${d.x}, ${d.y})`);
        });
    }
    
    async handleNodeClick(event, d) {
        event.stopPropagation();
        
        if (event.shiftKey) {
            // Expand node on shift+click
            await this.loadSubgraph(d.id, 1);
        } else {
            // Select node and show details
            this.selectNode(d.id);
        }
    }
    
    async selectNode(nodeId) {
        this.selectedNode = nodeId;
        
        // Update visual selection
        this.nodeGroup.selectAll('.node')
            .classed('selected', d => d.id === nodeId);
        
        // Load and display node details
        try {
            const response = await fetch(`/api/kg/node/${nodeId}`);
            const details = await response.json();
            
            let detailsHtml = `
                <div class="detail-label">ID:</div>
                <div class="detail-value">${details.id}</div>
                <div class="detail-label">Label:</div>
                <div class="detail-value">${details.label}</div>
                <div class="detail-label">Type:</div>
                <div class="detail-value">${details.kind}</div>
            `;
            
            if (details.description) {
                detailsHtml += `
                    <div class="detail-label">Description:</div>
                    <div class="detail-value">${details.description}</div>
                `;
            }
            
            if (details.in_degree !== undefined) {
                detailsHtml += `
                    <div class="detail-label">Connections:</div>
                    <div class="detail-value">In: ${details.in_degree}, Out: ${details.out_degree}</div>
                `;
            }
            
            if (details.related_concepts && details.related_concepts.length > 0) {
                detailsHtml += `
                    <div class="detail-label">Related Concepts:</div>
                    <div class="related-list">
                        ${details.related_concepts.map(rc => 
                            `<div class="related-item" onclick="kgViz.loadSubgraph('${rc.id}')">${rc.label} (${rc.relationship})</div>`
                        ).join('')}
                    </div>
                `;
            }
            
            document.getElementById('nodeDetails').innerHTML = detailsHtml;
        } catch (error) {
            console.error('Error loading node details:', error);
        }
    }
    
    async search() {
        const query = document.getElementById('searchInput').value.trim();
        if (!query) return;
        
        try {
            const response = await fetch('/api/kg/search', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query: query, limit: 10 })
            });
            
            const data = await response.json();
            
            const resultsHtml = data.results.map(result => `
                <div class="search-result" onclick="kgViz.loadSubgraph('${result.id}')">
                    <div class="search-result-label">${result.label}</div>
                    <div class="search-result-kind">${result.kind}</div>
                </div>
            `).join('');
            
            document.getElementById('searchResults').innerHTML = resultsHtml || '<div>No results found</div>';
            
            // Cache search results for path finding
            data.results.forEach(r => {
                this.searchCache[r.label.toLowerCase()] = r.id;
            });
        } catch (error) {
            console.error('Error searching:', error);
        }
    }
    
    async findPath() {
        const sourceLabel = document.getElementById('sourceInput').value.trim();
        const targetLabel = document.getElementById('targetInput').value.trim();
        
        if (!sourceLabel || !targetLabel) return;
        
        // Try to find IDs from cache or current nodes
        let sourceId = this.searchCache[sourceLabel.toLowerCase()];
        let targetId = this.searchCache[targetLabel.toLowerCase()];
        
        // If not in cache, search in current nodes
        if (!sourceId) {
            const sourceNode = this.nodes.find(n => n.label.toLowerCase() === sourceLabel.toLowerCase());
            if (sourceNode) sourceId = sourceNode.id;
        }
        
        if (!targetId) {
            const targetNode = this.nodes.find(n => n.label.toLowerCase() === targetLabel.toLowerCase());
            if (targetNode) targetId = targetNode.id;
        }
        
        if (!sourceId || !targetId) {
            alert('Please search for the concepts first to find their IDs');
            return;
        }
        
        try {
            const response = await fetch('/api/kg/path', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    source_id: sourceId,
                    target_id: targetId,
                    max_length: 5
                })
            });
            
            const data = await response.json();
            
            if (data.path_exists) {
                // Add path nodes and edges to graph
                this.updateData(data.path_nodes, data.path_edges);
                
                // Highlight path
                const pathNodeIds = new Set(data.path_nodes.map(n => n.id));
                const pathEdges = new Set(data.path_edges.map(e => `${e.source}-${e.target}`));
                
                this.nodeGroup.selectAll('.node')
                    .classed('path-node', d => pathNodeIds.has(d.id));
                
                this.linkGroup.selectAll('.link')
                    .classed('path-link', d => {
                        const linkId = `${d.source.id || d.source}-${d.target.id || d.target}`;
                        return pathEdges.has(linkId);
                    });
                
                alert(`Path found! Length: ${data.length} hops`);
            } else {
                alert(data.message);
            }
        } catch (error) {
            console.error('Error finding path:', error);
        }
    }
    
    toggleLabels() {
        this.showLabels = !this.showLabels;
        this.nodeGroup.selectAll('.node-label')
            .style('display', this.showLabels ? 'block' : 'none');
    }
    
    resetView() {
        // Clear highlighting
        this.nodeGroup.selectAll('.node')
            .classed('selected', false)
            .classed('path-node', false);
        
        this.linkGroup.selectAll('.link')
            .classed('path-link', false);
        
        // Reset zoom
        this.svg.transition()
            .duration(750)
            .call(d3.zoom().transform, d3.zoomIdentity);
        
        // Clear selections
        this.selectedNode = null;
        document.getElementById('nodeDetails').innerHTML = '';
    }
    
    showTooltip(event, text) {
        const tooltip = document.getElementById('tooltip');
        tooltip.textContent = text;
        tooltip.style.left = (event.pageX + 10) + 'px';
        tooltip.style.top = (event.pageY - 10) + 'px';
        tooltip.classList.add('show');
    }
    
    showNodeTooltip(event, d) {
        let text = `${d.label}\n${d.kind}`;
        if (d.description) {
            text += `\n${d.description.substring(0, 100)}...`;
        }
        this.showTooltip(event, text);
    }
    
    hideTooltip() {
        document.getElementById('tooltip').classList.remove('show');
    }
    
    dragStarted(event, d) {
        if (!event.active) this.simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
    }
    
    dragged(event, d) {
        d.fx = event.x;
        d.fy = event.y;
    }
    
    dragEnded(event, d) {
        if (!event.active) this.simulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
    }
    
    handleResize() {
        this.width = window.innerWidth;
        this.height = window.innerHeight;
        
        this.svg
            .attr('width', this.width)
            .attr('height', this.height);
        
        this.simulation.force('center', d3.forceCenter(this.width / 2, this.height / 2));
        this.simulation.alpha(0.3).restart();
    }
}

// Initialize visualization when page loads
let kgViz;
document.addEventListener('DOMContentLoaded', () => {
    kgViz = new KnowledgeGraphVisualization();
});