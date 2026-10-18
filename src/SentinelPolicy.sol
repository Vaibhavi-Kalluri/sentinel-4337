// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract SentinelPolicy {
    address public owner;

    mapping(address => bool) public allowedTargets;
    mapping(bytes4 => bool) public blockedSelectors;

    event TargetPermissionUpdated(address indexed target, bool allowed);
    event SelectorPermissionUpdated(bytes4 indexed selector, bool blocked);

    error Unauthorized();
    error TargetNotAllowed(address target);
    error SelectorBlocked(bytes4 selector);

    modifier onlyOwner() {
        if (msg.sender != owner) revert Unauthorized();
        _;
    }

    constructor() {
        owner = msg.sender;
    }

    function setTargetAllowed(
        address target,
        bool allowed
    ) external onlyOwner {
        allowedTargets[target] = allowed;
        emit TargetPermissionUpdated(target, allowed);
    }

    function setSelectorBlocked(
        bytes4 selector,
        bool blocked
    ) external onlyOwner {
        blockedSelectors[selector] = blocked;
        emit SelectorPermissionUpdated(selector, blocked);
    }

    function checkOperation(
        address target,
        bytes calldata callData
    ) external view returns (bool allowed, string memory reason) {
        if (!allowedTargets[target]) {
            return (false, "TARGET_NOT_ALLOWED");
        }

        if (callData.length < 4) {
            return (false, "INVALID_CALLDATA");
        }

        bytes4 selector;

        assembly {
            selector := calldataload(callData.offset)
        }

        if (blockedSelectors[selector]) {
            return (false, "SELECTOR_BLOCKED");
        }

        return (true, "POLICY_APPROVED");
    }
}
