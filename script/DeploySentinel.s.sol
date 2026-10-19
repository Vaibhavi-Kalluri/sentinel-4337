// SPDX-License-Identifier: MIT
pragma solidity ^0.8.28;

import {Script} from "forge-std/Script.sol";
import {EntryPoint} from "account-abstraction/core/EntryPoint.sol";
import {IEntryPoint} from "account-abstraction/interfaces/IEntryPoint.sol";

import {SentinelPolicy} from "../src/SentinelPolicy.sol";
import {SentinelAccountFactory} from "../src/SentinelAccountFactory.sol";
import {SentinelTarget} from "../src/SentinelTarget.sol";

contract DeploySentinel is Script {
    function run()
        external
        returns (
            SentinelPolicy policy,
            SentinelAccountFactory factory,
            SentinelTarget target
        )
    {
        uint256 deployerPrivateKey = vm.envUint("PRIVATE_KEY");

        vm.startBroadcast(deployerPrivateKey);

        // Deploy the on-chain policy.
        policy = new SentinelPolicy();

        // Reuse the existing EntryPoint address supplied through the environment.
        address entryPointAddress = vm.envAddress("ENTRY_POINT");

        factory = new SentinelAccountFactory(
            IEntryPoint(entryPointAddress),
            policy
        );

        // Deploy a fresh target for this Sentinel deployment.
        target = new SentinelTarget();

        // Allow the target for policy-approved operations.
        policy.setTargetAllowed(address(target), true);

        vm.stopBroadcast();
    }
}
