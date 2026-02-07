// ============================================================================
// linked_list.s - ARM64 Linked List Traversal (Memory Latency Test)
// Chases pointers through a randomized linked list to expose memory latency
// ============================================================================
.global _linked_list_sum_asm
.text
.balign 16

// int linked_list_sum_asm(Node* head)
// x0 = head pointer
// Returns: sum of all node values in w0
_linked_list_sum_asm:
    mov     w1, #0              // sum = 0
    
.loop_list:
    cbz     x0, .end_list       // if (current == nullptr) goto end
    
    // Prefetch next node (helps a bit but can't overcome latency)
    ldr     x2, [x0, #0]        // x2 = current->next
    prfm    pldl1keep, [x2]     // Prefetch next node
    
    // Load value (offset 8 from Node* due to pointer being first)
    ldr     w3, [x0, #8]        // w3 = current->value
    add     w1, w1, w3          // sum += value
    
    // Move to next
    mov     x0, x2              // current = next
    
    b       .loop_list

.end_list:
    mov     w0, w1              // return sum
    ret

