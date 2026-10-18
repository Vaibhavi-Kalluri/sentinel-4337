// SPDX-License-Identifier: MIT
pragma solidity ^0.8.28;

import {SimpleAccount} from "account-abstraction/accounts/SimpleAccount.sol";
import {IEntryPoint} from "account-abstraction/interfaces/IEntryPoint.sol";
import {Exec} from "account-abstraction/utils/Exec.sol";
import {SentinelPolicy} from "./SentinelPolicy.sol";

contract SentinelAccount is SimpleAccount {
    SentinelPolicy public immutable sentinelPolicy;

    error OperationBlocked(address target, string reason);

    constructor(
        IEntryPoint anEntryPoint,
        SentinelPolicy policy_
    ) SimpleAccount(anEntryPoint) {
        require(address(policy_) != address(0), "Invalid policy");
        sentinelPolicy = policy_;
    }

    function execute(
        address target,
        uint256 value,
        bytes calldata data
    ) external override {
        _requireForExecute();

        (bool allowed, string memory reason) =
            sentinelPolicy.checkOperation(target, data);

        if (!allowed) {
            revert OperationBlocked(target, reason);
        }

        bool success = Exec.call(target, value, data, gasleft());

        if (!success) {
            Exec.revertWithReturnData();
        }
    }

    function executeBatch(
        Call[] calldata calls
    ) external override {
        _requireForExecute();

        // Validate every call before executing any call.
        for (uint256 i = 0; i < calls.length; i++) {
            (bool allowed, string memory reason) =
                sentinelPolicy.checkOperation(
                    calls[i].target,
                    calls[i].data
                );

            if (!allowed) {
                revert OperationBlocked(calls[i].target, reason);
            }
        }

        // Execute only after the entire batch passes policy checks.
        for (uint256 i = 0; i < calls.length; i++) {
            bool success = Exec.call(
                calls[i].target,
                calls[i].value,
                calls[i].data,
                gasleft()
            );

            if (!success) {
                Exec.revertWithReturnData();
            }
        }
    }
}
